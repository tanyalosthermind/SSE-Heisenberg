/*
Source: 1101.3281 Sandvik
SSE method for the nearest-neighbor antiferromagnetic (J>0) S=1/2 Heisenberg model on a square lattice with PBC. J = 1

The Hamiltonian reads
    H = J sum_bons SxSx + SySy + SzSz
      = N_bonds/4 + sum_bonds (SxSx+SySy)  - (1/4 - SzSz)
      = N_bonds/4 + sum_b (H^off_b - H^diag_b)
The constant 1/4 ensures H^diag_b >= 0 (and also H^off_b >= 0): both types of operators (diagonal and off-diagonal)
can act only on anti-parallel spins, i.e., any operator acting on a pair of parallel spins leads to a configuration
that does not contribute to the partition function.

We expand the partition function as
    Z = tr(exp(-beta H) ~= tr(exp(beta sum_b (H^diag_b - H^off_b)))
      = sum_{|alpha>} <alpha| sum_n 1/n! (beta sum_b (H^diag_b - H^off_b))^n |alpha>
      = sum_{|alpha>} sum_n sum_{a(p),b(p) for p=0,...,n-1} beta^n/n! (-1)^{n_offdiag} <alpha| prod_{p=0}^{M-1} H^a(p)_b(p)|alpha>
      = sum_{|alpha>} sum_{a(p),b(p) for p=0,...,M-1} beta^n (M-n)!/M! <alpha| prod_{p=0}^{M-1} H^a(p)_b(p)|alpha>

The (-1)^{n_offdiag} vanishes, because the number of offdiagonal terms needs to be even on bipartite lattices.
Looking at the last line of the above equations, we need to sample spin configurations {|alpha>}
and "operator strings" {a(p),b(p)} of a fixed length `M`. This length `M` is chosen large enough that we actually never
encounter configurations with n=M non-identity operators.
Here, a(p) = identity, diag, offdiag, and b(p) labels bonds. In the code, they are combined into a single number
    op = {-1        for the identity operator
         {2*b(p)    for an H^diag_b(p)
         {2*b(p)+1  for an H^off_b(p)

To get a(p), we can check op == -1 (identity?) and otherwise mod(op, 2) == 0 (diagonal/offdiagonal?)
and use b(p) = op // 2  (with integer division rounding down).
The spin configuration |alpha> is given by a 1D array with values +-1.
Sites are enumerated, bonds are given by two numbers specifying which sites they connect --
this separates the geometry of the lattice completely from the implementation of updates.

*/

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <exception>
#include <string>
#include <cstdlib>      // std::rand, std::srand
#include <random>
#include <stdexcept> // error exception
//#include <experimental/random> // std::experimental::randint

using namespace std;

// random generator function:
int myrandom (int i) { return std::rand()%i;}
random_device rd;     // Only used once to initialise (seed) engine
mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
//uniform_int_distribution<int> uni(0,1); // Guaranteed unbiased
uniform_real_distribution<> dis(0., 1.0);
int random(int low, int high)
{
    uniform_int_distribution<> dist(low, high);
    return dist(rng);
}

// mean of vector
template<typename T>
double getAverage(std::vector<T> const& v) {
    if (v.empty()) {
        return 0;
    }
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}
// standard deviation
template<typename T1>
double getStd(std::vector<T1> const& v) {
    double mean;
    if (v.empty()) {
        mean = 0;
    }
    mean = accumulate(v.begin(), v.end(), 0.0) / v.size();
    vector<double> diff(v.size());
    transform(v.begin(), v.end(), diff.begin(),
    bind2nd(minus<double>(), mean));
    double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = sqrt(sq_sum / v.size());
    return stdev;
}


// structure to store configuration
struct configuration{
    vector <int> spins;
    vector <int> op_string;
    vector< vector <int> > bonds;
};

// structure to store vertices
struct vertex{
    vector <int> vertex_list;
    vector <int> first_vertex_at_site;
    vector <int> last_vertex_at_site;
};

int site(int x, int y, int Lx, int Ly);
configuration init_SSE_square(int Lx, int Ly);
int diagonal_update(vector <int> & spins, vector <int> & op_string, vector< vector <int> > & bonds, double beta);
void loop_update(vector <int> & spins, vector <int> & op_string, vector< vector <int> > & bonds);
vertex create_linked_vertex_list(vector <int> & spins, vector <int> & op_string, vector< vector <int> > & bonds);
void flip_loops(vector <int> & spins, vector <int> & op_string, vector <int> & vertex_list, vector <int> & first_vertex_at_site);
vector<int> thermalize(vector <int> & spins, vector <int> & op_string, vector< vector <int> > & bonds, double beta, int n_updates_warmup);
vector<int> measure(vector <int> & spins, vector <int> & op_string, vector< vector <int> > & bonds, double beta, int n_updates_measure);
vector< vector <double> > run_simulation(int Lx, int Ly, vector <double> & betas, int n_updates_measure, int n_bins);

int main() {
    //const int Lx = 4;
    //const int Ly = 4;

    double beta1 = 5.;
    //cout << "beta = " << beta << endl;
    double beta2 = 25.;

    int n_updates_measure = 10000;
    int n_bins = 10;

    vector <double> betas;
    betas.push_back(beta1);
    betas.push_back(beta2);

    vector<int> Ls; // sizes have to be even for this lattice to be bipartite.
    Ls.push_back(4);
    Ls.push_back(6);
    Ls.push_back(8);
    Ls.push_back(10);
    Ls.push_back(12);
    Ls.push_back(14);
    Ls.push_back(16);


    for (int i = 0; i < Ls.size(); i++){
        vector< vector <double> > Es_Eerrs = run_simulation(Ls[i], Ls[i], betas, n_updates_measure, n_bins);
        cout << "L = " << Ls[i] << "; Energy per site = " <<  Es_Eerrs[0][0] << " error = " << Es_Eerrs[0][1]  << " at T = " << 1/beta1 << endl;
        cout << "L = " << Ls[i] << "; Energy per site = " <<  Es_Eerrs[1][0] << " error = " << Es_Eerrs[1][1]  << " at T = " << 1/beta2 << endl;
    }
    //vector< vector <double> > Es_Eerrs = run_simulation(Lx, Ly, betas, n_updates_measure, n_bins);
    //cout << "L = " << Lx << "; Energy per site = " <<  Es_Eerrs[0][0] << " error = " << Es_Eerrs[0][1]  << endl;
    //cout << "L = " << Lx << "; Energy per site = " <<  Es_Eerrs[1][0] << " error = " << Es_Eerrs[1][1]  << endl;



  return 0;
}



// functions

int site(int x, int y, int Lx, int Ly) {
        // Defines a numbering of the sites, given positions x and y.
        // label the lattice sites (x,y) by a single integer number n = y * Lx + x = 0, 1, ..., N-1

        int result = 0;
        result = y * Lx + x;
        return result;
    }

configuration init_SSE_square(int Lx, int Ly) {
    //Initialize a starting configuration on a 2D square lattice.

    int N = Lx * Ly;
    configuration Config;

    //initialize spins randomly with numbers +1 or -1, but the average magnetization is 0.
    vector <int> range_N;
    for (int i = 0; i < N; i++) {
        range_N.push_back(i);
    }
    random_shuffle(range_N.begin(), range_N.end());// myrandom);
    for (int i = 0; i < range_N.size(); i++) {
        Config.spins.push_back(2* (range_N[i] % 2) - 1);
    }
    for (int i = 0; i < 10; i++) { // size of operator string is 10: set here
        Config.op_string.push_back(-1);
    }

    /*Bond notation. For the 2D square lattice with periodic boundary conditions B=2N.
    This is a vector that contains the indicies of each bond (n,m) of the square lattice with PBS once.
    The vector has shape (2N, 2).
    This list is the only information on the lattice geometry needed in the sampling of configurations.
    */
    for (int x0 = 0; x0 < Lx; x0++){
        for (int y0 = 0; y0 < Ly; y0++){
            int s0 = site(x0, y0, Lx, Ly);
            int s1 = site(((x0+1) % Lx), y0, Lx, Ly); // bond to the right
            //cout << s0 << " " << s1 << endl;
            int arr[] = {s0, s1};
            vector<int> s01 (arr, arr + sizeof(arr) / sizeof(arr[0]) );
            Config.bonds.push_back(s01);
            int s2 = site(x0, ((y0+1) % Ly), Lx, Ly);
            int arr2[] = {s0, s2};
            vector<int> s02 (arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]) );
            Config.bonds.push_back(s02);
        }
    }


    return Config;
}

int diagonal_update(vector<int> & spins, vector<int> & op_string, vector< vector <int> > & bonds, double beta){
    /* Perform the diagonal update: insert or remove diagonal operators into/from the op_string.
    The purpose of the diagonal update is to change the number n of hamiltonian operators in the sequence.
    -> to substitute unit operators opstring[p]=-1 by diagonal operators opstring[p]=2*b and vice versa.
    While one can always substitute a diagonal operator by a unit operator and obtain a new valid configuration,
    the insertion of a diagonal operator acting on a bond b requires that the spins at this bond are in an antiparallel
     state in the propagated state |&alpha(p)>.
     Off-diagonal operators cannot be inserted or removed one-by one and they are thus left unchanged in this update.
    */
    int n_bonds = bonds.size();
    int M = op_string.size();

    // count the number of non-identity operators
    int n = 0;
    for (int i = 0; i < M; i++){
        if (op_string[i] != -1){
            n++;
        }
    }

    // calculate ratio of acceptance probabilities for insert/remove  n <-> n+1
    // <alpha|Hdiag|alpha> = 1/4 + <alpha |SzSz|alpha> = 0.5 for antiparallel spins
    double prob_ratio = 0.5 * beta * n_bonds; //(M-n) , but the latter depends on n which still changes
    int op = 0;
    for (int p = 0; p < M; p++){ // go through the operator string
        op = op_string[p];
        if (op == -1){ // identity: propose to insert a new operator
            int b = 0 + (rand() % static_cast<int>(n_bonds - 1 - 0 + 1));
            if (spins[bonds[b][0]] != spins[bonds[b][1]]){
                //can only insert if the two spins are anti-parallel!
                double prob1 = prob_ratio / (M - n);
                double r = ((double) rand() / (RAND_MAX));
                if (r < prob1){ // (metropolis-like)
                    // insert a diagonal operator
                    op_string[p] = 2 * b;
                    n+=1;
                }
            }
        } else if ((op % 2) == 0) {//diagonal operator: propose to remove
            double prob2 = 1/prob_ratio * (M-n+1); //n-1 = number operators after removal = n in above formula
            double r2 = ((double) rand() / (RAND_MAX));
            if (r2 < prob2){
                // remove diagonal operator
                op_string[p] = - 1;
                n-=1;
            }
        } else { //  encountered operator is an off-diagonal one! -> spins are flipped
            int bb = op / 2;
            // H^off ~= (S+S- + S-S+) = spin flip on both sites for antiparallel spins.
            // (We never have configurations with operators acting on parallel spins!)
            spins[bonds[bb][0]] = - spins[bonds[bb][0]];
            spins[bonds[bb][1]] = - spins[bonds[bb][1]];
        }
    }
    /* When we have completed the diagonal update at the last position p=M in the string, the stored spins
    have been propagated back to the original state (because of the periodicity |&alpha(M)>=|&alpha(0)>)
    and we move on to the next type of update.
    */
    return n;
}

void loop_update(vector<int> & spins, vector<int> & op_string, vector< vector <int> > & bonds){
    /* Perform the offdiagonal update: construct loops and flip each of them with prob. 0.5.
    PBC implies that each spin must be flipped an even number of times (or not at all) during the propagation from p=1 to p=N.
    This in turn implies that there has to be an even number of off-diagonal operators in the string. */
    // create the loops
    vertex Vertex = create_linked_vertex_list(spins, op_string, bonds);
    // and flip them
    flip_loops(spins, op_string, Vertex.vertex_list, Vertex.first_vertex_at_site);
}


vertex create_linked_vertex_list(vector<int> & spins, vector<int> & op_string, vector< vector <int> > & bonds){
    /* This linked vertex representation captures directly the way the operators are "connected", i.e.,
    using it one can directly jump from an operator at position p, which acts on two spins i(b(p)) and j(b(p))
    (in the program bonds[0,opstring[p]/2] and bonds[1,opstring[p]/2]) to the next or previous operator acting
    on one of these spins, without having to conduct a tedious search in the list opstring[].
    Each bond operator acts on two "in" spins and the result of this operation is two "out" spins.
    In the linked vertex representation an operator is represented by a vertex with four legs.
    Each leg has a corresponding spin state. There are four allowed vertices.
    So if we simply enumerate all the vertices in the operator
    string, we get v0 = 4*p, v1=4*p+1, v2=4*p+2, v4=4*p+3 for the vertices
        v0  v1
         |--|
         |Op|  <-- op_string[p]
         |--|
        v2  v3
    In this function, we set the entries of the `vertex_list` for any (vertically)
    connected pair `v, w` (i.e. vertical parts of the loops) we have
    ``v = vertex_list[w]`` and ``w = vertex_list[v]``.
    Later on, an entry -1 indicates that the loop along this connection was flipped;
    an entry -2 indices that the loop was visited and proposed to flip, but the flip was rejected.
    Identity operators are completely ignored for the connections, its vertices are directly
    marked with a -2.
    The returned array `first_vertex_at_site` contains the first vertex encountered at each site,
    entries -1 indicate that there is no (non-identity) operator acting on that site.
    */
    vertex Vertex;
    //Given a configuration, construct a linked list between vertices defining the loops.
    //The linked list is a temporary data structure, to be used in one of the configuration updates.
    //It is constructed before this update and destroyed after.
    int n_sites = spins.size();
    int M = op_string.size();
    for (int i = 0; i < 4*M; i++){
        Vertex.vertex_list.push_back(0);
    }
    for (int i = 0; i < n_sites; i++){
        Vertex.first_vertex_at_site.push_back(-1); //-1 = no vertex found (yet)
        Vertex.last_vertex_at_site.push_back(-1);
    }
    // iterate over all operators
    int v0, v1, v2, v3, v4;
    int op;
    for (int p = 0; p < M; p++){
        v0 = p*4; //left incoming vertex
        v1 = v0 + 1; // right incoming vertex
        op = op_string[p];
        if (op == -1){ //identity operator
            // ignore it for constructing/flipping loops: mark as visited
            Vertex.vertex_list.at(v0) = -2;
            Vertex.vertex_list.at(v0+1) = -2;
            Vertex.vertex_list.at(v0+2) = -2;
            Vertex.vertex_list.at(v0+3) = -2;
            //for (int i = 0; i <= 4; i++){
                    //int vv = v0+i;
                    //Vertex.vertex_list.at(vv) = -2;
            //        Vertex.vertex_list[v0+i] = -2;
            //}
        }
        else {
            int b = op / 2;
            int s0 = bonds[b][0];
            int s1 = bonds[b][1];
            v2 = Vertex.last_vertex_at_site[s0];
            v3 = Vertex.last_vertex_at_site[s1];
            if (v2 == -1){ //no operator encountered at this site before
                Vertex.first_vertex_at_site[s0] = v0;
            }
            else { //encountered an operator at this vertex before -> create link
                Vertex.vertex_list[v2] = v0;
                Vertex.vertex_list[v0] = v2;
            }
            if (v3 == -1){ // similar for the other site
                Vertex.first_vertex_at_site[s1] = v1;
            }
            else {
                Vertex.vertex_list[v3] = v1;
                Vertex.vertex_list[v1] = v3;
            }
            Vertex.last_vertex_at_site[s0] = v0 + 2;  // left outgoing vertex of op
            Vertex.last_vertex_at_site[s1] = v0 + 3;  // right outgoing vertex of op
        }
    }
    // now we need to connect vertices between top and bottom
    for (int s00 = 0; s00 < n_sites; s00++){
        v0 = Vertex.first_vertex_at_site[s00];
        if (v0 != -1){  // there is an operator acting on that site -> create link
            v1 = Vertex.last_vertex_at_site[s00];
            Vertex.vertex_list[v1] = v0;
            Vertex.vertex_list[v0] = v1;
        }
    }

    return Vertex;
}

void flip_loops(vector<int> & spins, vector<int> & op_string, vector<int> & vertex_list, vector<int> & first_vertex_at_site){
    /*  construct a loop connecting vertex legs
    Given the vertex_list, flip each loop with prob. 0.5 (analog to Swendsen-Wang cluster update).
    Once we have the vertex list, we can go through all the vertices and flip each loop with
    probability 0.5. When we propose to flip a loop, we go through it and mark it as flipped (-1)
    or visited (-2) in the vertex list to avoid a secend proposal to flip it.
    Note that for an integer number `i`, the operation ``i ^ 1`` gives i+1 or i-1 depending on
    whether `i` is even or odd: it flips 0<->1, 2<->3, 4<->5, ...
    This is used to switch between diagonal/offdiagonal operators in the operator string when
    flipping a loop, and to propagate the open end of the loop vertically between vertices
    v0<->v1, v2<->v3 of the operators.
    */
    int n_sites = spins.size();
    int M = op_string.size();
    //iterate over all possible beginnings of loops
    // (step 2: v0+1 belongs to the same loop as v0)
    for (int v0 = 0; v0 < 4*M; v0++){
        if (v0 % 2 == 0){// STEP 2 !!!!
            if (vertex_list[v0] < 0){ //marked: we've visited the loop starting here before.
                continue;
            }
            int v1 = v0; //  we move v1 as open end of the loop around until we come back to v0
            double r = ((double) rand() / (RAND_MAX));
            if (r < 0.5){
                // go through the loop and flip it
                while (true) {
                    int op = v1 / 4;
                    // flip diagonal/off-diagonal! adding 1 to an even number a and subtracting 1 from an odd a
                    op_string[op] = op_string[op] ^ 1;
                    vertex_list[v1] = -1;
                    int v2 = v1 ^ 1;
                    v1 = vertex_list[v2];
                    vertex_list[v2] = -1;
                    if (v1 == v0){ // loop closes
                        break;
                    }
                }
            }
            else {
                // don't flip the loop, but go through it to mark it as visited
                while (true){
                    vertex_list[v1] = -2;
                    int v2 = v1 ^ 1;
                    v1 = vertex_list[v2];
                    vertex_list[v2] = -2;
                    if (v1 == v0){
                        break;
                    }
                }
            }
        }
    }
    for (int s00 = 0; s00 < n_sites; s00++){
        if (first_vertex_at_site[s00] == -1){  // no operator acting on that site -> flip with p=0.5
            if (((double) rand() / (RAND_MAX)) < 0.5){
                spins[s00] = -spins[s00];
            }
        }
        else {  // there is an operator acting on that site
            if (vertex_list[first_vertex_at_site[s00]] == -1){  // did we flip the loop?
                spins[s00] = -spins[s00]; // then we also need to flip the spin
            }
        }
    }
/* a loop update can affect a very large number of operators and hence it can be expected to be
 more efficient than the pair update. */
}

vector<int> thermalize(vector<int> & spins, vector<int> & op_string, vector< vector <int> > & bonds, double beta, int n_updates_warmup){
    //cout << " thermalize " << endl;
    //Perform a lot of upates to thermalize, without measurements.
    if (beta == 0.){
            throw invalid_argument( "Simulation doesn't work for beta = 0" );
    }
    for (int i = 0; i < n_updates_warmup; i++){
        int n = diagonal_update(spins, op_string, bonds, beta);
        loop_update(spins, op_string, bonds);
        // check if we need to increase the length of op_string
        int M_old = op_string.size(); // len(op_string)
        int M_new = n + n / 3;
        if (M_new > M_old){
            op_string.resize(M_new, -1);
        }
    }
    return op_string;
}

vector<int> measure(vector <int> & spins, vector <int> & op_string, vector< vector <int> > & bonds, double beta, int n_updates_measure){
    //cout << " measure " << endl;
    // Perform a lot of updates with measurements.
    vector<int> ns;
    for (int i = 0; i < n_updates_measure; i++){
        int n = diagonal_update(spins, op_string, bonds, beta);
        loop_update(spins, op_string, bonds);
        ns.push_back(n);
    }
    return ns;
}


vector< vector <double> > run_simulation(int Lx, int Ly, vector <double> & betas, int n_updates_measure, int n_bins){
    //cout << " run_simulation " << endl;
    //A full simulation: initialize, thermalize and measure for various betas.
    configuration Config;
    Config = init_SSE_square(Lx, Ly);
    int n_sites = Config.spins.size();
    int n_bonds = Config.bonds.size();
    vector< vector <double> > Es_Eerrs;
    for (int i = 0; i < betas.size(); i++){
        cout << "beta = " << betas[i] << endl;
        vector <int> therm_config = thermalize(Config.spins, Config.op_string, Config.bonds, betas[i], n_updates_measure / 10);
        Config.op_string = therm_config;
        vector<double> Es;
        for (int j = 0; j < n_bins; j++){
            vector<int> ns = measure(Config.spins, Config.op_string, Config.bonds, betas[i], n_updates_measure);
            // energy per site
            double ns_mean = getAverage(ns);
            double E = ( - ns_mean / betas[i] + 0.25 * n_bonds) / n_sites;
            Es.push_back(E);
        }
        double E_mean = getAverage(Es);
        double E_error = getStd(Es) / sqrt(n_bins);
        double arr[] = {E_mean, E_error};
        vector<double> E_pair (arr, arr + sizeof(arr) / sizeof(arr[0]) );
        Es_Eerrs.push_back(E_pair);
    }
    return Es_Eerrs;

}
