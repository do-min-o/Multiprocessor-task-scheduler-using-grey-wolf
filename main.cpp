#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

// Define the graph structure
struct Task {
  int id;
  int weight;
  vector<int> dependencies;
  vector<int> communicationCosts; // Communication costs with other tasks
};

// Define the wolf structure
struct Wolf {
    vector<long double> solution; // Assignment of tasks to processors
    int makespan; // Personal best fitness value
};

// Define parameters
// Initialize the tasks and their dependencies
// const vector<Task> TASKS = {{0, 2, {}, {0, 0, 0, 0, 0, 0, 0, 0, 0}},
//                        {1, 3, {0}, {4, 0, 0, 0, 0, 0, 0, 0, 0}},
//                        {2, 3, {0}, {1, 0, 0, 0, 0, 0, 0, 0, 0}},
//                        {3, 4, {0}, {1, 0, 0, 0, 0, 0, 0, 0, 0}},
//                        {4, 5, {0}, {1, 0, 0, 1, 0, 1, 0, 0, 0}},
//                        {5, 4, {1}, {0, 1, 0, 0, 0, 0, 0, 0, 0}},
//                        {6, 4, {0, 1, 2}, {20, 5, 5, 0, 0, 0, 0, 0, 0}},
//                        {7, 4, {1, 2, 3, 4}, {0, 5, 1, 1, 10, 0, 0, 0, 0}},
//                        {8, 1, {5, 6, 7}, {0, 0, 0, 0, 0, 10, 10, 10, 0}}};

// const vector<Task> TASKS = {
//     {0, 5, {}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {1, 4, {0}, {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {2, 7, {0}, {5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {3, 3, {0}, {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {4, 8, {0}, {5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {5, 4, {1}, {0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {6, 7, {2, 5}, {0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {7, 9, {3, 5}, {0, 0, 0, 6, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {8, 4, {4, 5}, {0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
//     {9, 7, {6}, {0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0}},
//     {10, 4, {7, 9}, {0, 0, 0, 0, 0, 0, 0, 6, 0, 6, 0, 0, 0, 0}},
//     {11, 8, {8, 9}, {0, 0, 0, 0, 0, 0, 0, 0, 7, 5, 0, 0, 0, 0}},
//     {12, 4, {10}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0}},
//     {13, 6, {11, 12}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0}}
// };

const vector<Task> TASKS = {
    {0,5,{},{0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
    {1,4,{0},{10,0,0,0,0,0,0,0,0,0,0,0,0,0}},
    {2,7,{0},{11,0,0,0,0,0,0,0,0,0,0,0,0,0}},
    {3,3,{0},{12,0,0,0,0,0,0,0,0,0,0,0,0,0}},
    {4,8,{0},{9,0,0,0,0,0,0,0,0,0,0,0,0,0}},
    {5,4,{1},{0,5,0,0,0,0,0,0,0,0,0,0,0,0}},
    {6,7,{2,5},{0,0,12,0,0,8,0,0,0,0,0,0,0,0}},
    {7,9,{3,5},{0,0,0,13,0,11,0,0,0,0,0,0,0,0}},
    {8,4,{4,5},{0,0,0,0,14,10,0,0,0,0,0,0,0,0}},
    {9,7,{6},{0,0,0,0,0,0,12,0,0,0,0,0,0,0}},
    {10,4,{7,9},{0,0,0,0,0,0,0,13,0,13,0,0,0,0}},
    {11,8,{8,9},{0,0,0,0,0,0,0,0,14,12,0,0,0,0}},
    {12,4,{10},{0,0,0,0,0,0,0,0,0,0,15,0,0,0}},
    {13,6,{11,12},{0,0,0,0,0,0,0,0,0,0,0,16,15,0}}
};

const int N = TASKS.size(); // Number of tasks
const int M = 4;        // Number of processors
const int MAX_ITER = 100; // Maximum number of iterations
const int NUM_OF_WOLVES = 1000; // Number of wolves in PSO
const long double MIN_POS = 0, MAX_POS = N-1;
const long double a = 2.0;
vector<Wolf> wolves;

Wolf alpha, beta2, delta;
// Evaluate makespan of a solution
// priority: priority[i] is the priority of i'th task in the order of tasks
int evaluateMakespan(const vector<long double> &priority) {
    priority_queue<pair<int,int>> pq;
    vector<int> deps_left(N, 0);
    vector<vector<int>> dependents(N);
    for(int i=0;i<N;i++) {
        if(TASKS[i].dependencies.empty()) {
            pq.push({priority[i], i});
            continue;
        }
        deps_left[i] = TASKS[i].dependencies.size();
        for(int dependency:TASKS[i].dependencies) {
            dependents[dependency].push_back(i);
        }
    }
    vector<int> finishTimesForProcessor(M, 0), finishProc(N, -1), finishTimesForTask(N, -1); // Finish times for each processor
    while(!pq.empty()) {
        int task_id = pq.top().second;
        pq.pop();
        int nax = -1e9, nax2 = -1e9, nax_processor_id = -1;
        for(int dependency:TASKS[task_id].dependencies) {
            int comm_time = finishTimesForTask[dependency] + TASKS[task_id].communicationCosts[dependency];
            if(comm_time >= nax) {
                nax2 = nax;
                nax = comm_time;
                nax_processor_id = finishProc[dependency];
            } else nax2 = max(nax2, comm_time);
        }

        // find best processor
        int best_start_time = 1e9, best_processor = -1;
        for(int i=0;i<M;i++) {
            int start_time;
            if(i == nax_processor_id) {
                start_time = max(finishTimesForProcessor[i], nax2);
            } else start_time = max(finishTimesForProcessor[i], nax);
            if(start_time < best_start_time) {
                best_start_time = start_time;
                best_processor = i;
            }
        }
        finishProc[task_id] = best_processor;
        finishTimesForProcessor[best_processor] = best_start_time + TASKS[task_id].weight;
        finishTimesForTask[task_id] = best_start_time + TASKS[task_id].weight;
        events.push({-finishTimesForProcessor[best_processor], task_id});
        for(int dependent: dependents[task_id]) {
            deps_left[dependent]--;
            if(deps_left[dependent] == 0) pq.push({priority[dependent], dependent});
        }
    }

    if(*min_element(finishProc.begin(), finishProc.end()) == -1) {
        cout<<"some tasks weren't completed, cycle present" << endl;
        exit(-1);
    }
    // Maximum finish time among all processors is the makespan
    return *max_element(finishTimesForProcessor.begin(), finishTimesForProcessor.end());
}

long double PSO(int number_of_threads) {
    vector<Wolf> wolves(NUM_OF_WOLVES);
    vector<long double> global_best;
    int global_best_makespan = 1e9;

#pragma omp parallel for num_threads(number_of_threads) schedule(dynamic)
    for(Wolf &wolf:wolves) {
        wolf.solution.resize(N);
        for(int j=0;j<N;j++) {
            wolf.solution[j] = MIN_POS + (long double)rand()/RAND_MAX*(MAX_POS-MIN_POS);
        }
        wolf.makespan = evaluateMakespan(wolf.solution);
    }
    alpha = wolves[0];
    beta2 = wolves[0];
    delta = wolves[0];
    for(Wolf &wolf:wolves) {
        if(wolf.makespan < global_best_makespan) {
            global_best_makespan = wolf.makespan;
        }
    }
    for(int iter=1;iter<MAX_ITER;iter++) {
        // update each wolf, can be parallelized
#pragma omp parallel for num_threads(number_of_threads) schedule(dynamic)
        for (const Wolf& wolf : wolves) {
            if (wolf.makespan < alpha.makespan) {
                delta = beta2;
                beta2 = alpha;
                alpha = wolf;
            } else if (wolf.makespan < beta2.makespan) {
                delta = beta2;
                beta2 = wolf;
            } else if (wolf.makespan < delta.makespan) {
                delta = wolf;
            }
        }

#pragma omp parallel for num_threads(number_of_threads) schedule(dynamic)
        for(Wolf &wolf:wolves) {
            // update velocity
            for(int i=0;i<N;i++) {
                long double r1 = (long double)rand() / RAND_MAX;
                long double r2 = (long double)rand() / RAND_MAX;
                long double A = ((MAX_ITER - iter)/(MAX_ITER))*a*(2 * r1 - 1);
                long double C = 2 * r2;

                // Update solution
                long double D_alpha = abs(C * alpha.solution[i] - wolf.solution[i]);
                long double D_beta2 = abs(C * beta2.solution[i] - wolf.solution[i]);
                long double D_delta = abs(C * delta.solution[i] - wolf.solution[i]);
                long double rand_factor = (long double)rand() / RAND_MAX;
                wolf.solution[i] = (alpha.solution[i] - A * D_alpha + beta2.solution[i] - A * D_beta2 + delta.solution[i] - A * D_delta + rand_factor) / 4;
            }
            wolf.makespan = evaluateMakespan(wolf.solution);
        }

#pragma omp parallel for num_threads(number_of_threads) schedule(dynamic)
        for(Wolf &wolf:wolves) {
            if(wolf.makespan < global_best_makespan) {
                global_best_makespan = wolf.makespan;
            }
        }
        cout << global_best_makespan << "\n";
    }
    return global_best_makespan;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    int number_of_threads = stoi(argv[1]);
    cout << "Lets begin" << endl;
    double start_time = omp_get_wtime();
    long double answer = PSO(number_of_threads);
    cout << answer << "\n";
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;
    return 0;
}