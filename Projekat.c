#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <string.h>

#define INF INT_MAX

typedef struct Edge {
    int to;
    int weight;
    struct Edge* next;
} Edge;

typedef struct {
    int N;
    Edge** adj;
} Graph;

Graph* create_graph(int N) {
    Graph* g = malloc(sizeof(Graph));
    g->N = N;
    g->adj = calloc(N, sizeof(Edge*));
    return g;
}

void add_edge(Graph* g, int u, int v, int w) {
    Edge* e = malloc(sizeof(Edge));
    e->to = v;
    e->weight = w;
    e->next = g->adj[u];
    g->adj[u] = e;
}

// Generisanje grafa
void generate_graph(Graph* g, float density) {
    int N = g->N;
    int max_edges = N * (N - 1);
    int target_edges = (int)(density * max_edges);

    srand((unsigned)time(NULL));

    int added = 0;
    while (added < target_edges) {
        int u = rand() % N;
        int v = rand() % N;
        if (u == v) continue;

        int exists = 0;
        for (Edge* e = g->adj[u]; e; e = e->next)
            if (e->to == v) { exists = 1; break; }
        if (exists) continue;

        int w = rand() % 101 - 50; // [-50,50]
        add_edge(g, u, v, w);
        added++;
    }
}

// DFS za check povezanosti u orientiranom grafu
void dfs(Graph* g, int v, int* vis) {
    vis[v] = 1;
    for (Edge* e = g->adj[v]; e; e = e->next)
        if (!vis[e->to]) dfs(g, e->to, vis);
}

// Prosta (ali ne optimalna) provjera jako jake povezanosti: je li za svaki par moguće ići u oba smjera?
int is_strongly_connected(Graph* g) {
    int N = g->N;
    int* vis = calloc(N, sizeof(int));
    dfs(g, 0, vis);
    for (int i = 0; i < N; i++) if (!vis[i]) { free(vis); return 0; }
    free(vis);

    Graph* gt = create_graph(N);
    for (int u = 0; u < N; u++)
        for (Edge* e = g->adj[u]; e; e = e->next)
            add_edge(gt, e->to, u, e->weight);

    vis = calloc(N, sizeof(int));
    dfs(gt, 0, vis);
    int ok = 1;
    for (int i = 0; i < N; i++) if (!vis[i]) { ok = 0; break; }
    free(vis);
    // oslobađanje gt adiate je prepušteno uprošćenom pristupu
    return ok;
}

// Pokušaj minimalnog dodavanja bridova da postane jako povezan — jednostavna heuristika
void make_strongly_connected(Graph* g) {
    int N = g->N;
    if (is_strongly_connected(g)) return;

    for (int u = 0; u < N; u++) {
        int has_in = 0, has_out = 0;
        for (int v = 0; v < N; v++) {
            if (u == v) continue;
            for (Edge* e = g->adj[u]; e; e = e->next)
                if (e->to == v) has_out = 1;
            for (Edge* e = g->adj[v]; e; e = e->next)
                if (e->to == u) has_in = 1;
        }
        if (!has_in) {
            int v = (u + 1) % N;
            add_edge(g, v, u, rand()%101 - 50);
        }
        if (!has_out) {
            int v = (u + 1) % N;
            add_edge(g, u, v, rand()%101 - 50);
        }
    }
}

// Dijkstra (bez negativnih ciklusa): Ako su teže negativne, ovakav pristup NE garantuje ispravnost u svim slučajevima!
void dijkstra(Graph* g, int src, int* dist) {
    int N = g->N;
    int* vis = calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) dist[i] = INF;
    dist[src] = 0;

    for (int it = 0; it < N; it++) {
        int u = -1, best = INF;
        for (int i = 0; i < N; i++)
            if (!vis[i] && dist[i] < best) { best = dist[i]; u = i; }
        if (u == -1) break;
        vis[u] = 1;
        for (Edge* e = g->adj[u]; e; e = e->next)
            if (dist[u] + e->weight < dist[e->to])
                dist[e->to] = dist[u] + e->weight;
    }
    free(vis);
}

// Oznake: -1 neoznačeno, 0 i 1 označeno
void random_label_nodes(int N, int* label) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) label[i] = -1;
    int count0 = N * 30 / 100;
    int count1 = N * 40 / 100;
    int assigned0 = 0, assigned1 = 0;

    while (assigned0 < count0) {
        int u = rand() % N;
        if (label[u] == -1) { label[u] = 0; assigned0++; }
    }
    while (assigned1 < count1) {
        int u = rand() % N;
        if (label[u] == -1) { label[u] = 1; assigned1++; }
    }
}

// KNN bez ažuriranja među neoznačenim
void knn_labeling(Graph* g, int* label, int k) {
    int N = g->N;
    int* newlabel = malloc(N * sizeof(int));
    memcpy(newlabel, label, N * sizeof(int));
    int* dist = malloc(N * sizeof(int));

    int *idx = malloc(N * sizeof(int));
    for (int u = 0; u < N; u++) {
        if (label[u] != -1) continue;
        dijkstra(g, u, dist);

        for (int i = 0; i < N; i++) idx[i] = i;
        // sortiraj po rastućoj udaljenosti
        for (int i = 0; i < N-1; i++)
            for (int j = i+1; j < N; j++)
                if (dist[idx[i]] > dist[idx[j]]) {
                    int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
                }

        int zeros = 0, ones = 0, taken = 0;
        for (int t = 0; t < N && taken < k; t++) {
            int v = idx[t];
            if (label[v] == -1 || dist[v] == INF) continue;
            if (label[v] == 0) zeros++;
            else if (label[v] == 1) ones++;
            taken++;
        }
        newlabel[u] = (ones > zeros) ? 1 : 0;
    }
    free(idx);
    memcpy(label, newlabel, N * sizeof(int));
    free(newlabel);
    free(dist);
}

// KNN sa progresivnim ažuriranjem
void knn_labeling_incremental(Graph* g, int* label, int k) {
    int N = g->N;
    int* dist = malloc(N * sizeof(int));
    int *idx = malloc(N * sizeof(int));
    for (int u = 0; u < N; u++) {
        if (label[u] != -1) continue;
        dijkstra(g, u, dist);

        for (int i = 0; i < N; i++) idx[i] = i;

        for (int i = 0; i < N-1; i++)
            for (int j = i+1; j < N; j++)
                if (dist[idx[i]] > dist[idx[j]]) {
                    int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
                }

        int zeros = 0, ones = 0, taken = 0;
        for (int t = 0; t < N && taken < k; t++) {
            int v = idx[t];
            if (label[v] == -1 || dist[v] == INF) continue;
            if (label[v] == 0) zeros++;
            else if (label[v] == 1) ones++;
            taken++;
        }
        label[u] = (ones > zeros) ? 1 : 0;
    }
    free(idx);
    free(dist);
}

// Mjerenje vremena
double measure_time(void (*fn)(Graph*, int*, int), Graph* g, int* label, int k) {
    clock_t start = clock();
    fn(g, label, k);
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

void compute_and_report(Graph* g, int k, int method_id) {
    int N = g->N;
    int* label = malloc(N * sizeof(int));
    random_label_nodes(N, label);
    int zeros_before = 0, ones_before = 0;
    for (int i = 0; i < N; i++) {
        if (label[i] == 0) zeros_before++;
        else if (label[i] == 1) ones_before++;
    }

    double t = 0;
    int* label_copy = malloc(N * sizeof(int));
    memcpy(label_copy, label, N * sizeof(int));

    if (method_id == 0) t = measure_time(knn_labeling, g, label_copy, k);
    else t = measure_time(knn_labeling_incremental, g, label_copy, k);

    int zeros_after = 0, ones_after = 0;
    for (int i = 0; i < N; i++) {
        if (label_copy[i] == 0) zeros_after++;
        else if (label_copy[i] == 1) ones_after++;
    }

    printf("Method %d, k=%d: time=%.3f s, zeros=%d, ones=%d (before: %d/%d)\n",
           method_id, k, t, zeros_after, ones_after, zeros_before, ones_before);

    free(label);
    free(label_copy);
}

// Returns 1 if negative cycle exists, 0 otherwise
int has_negative_cycle(Graph* g) {
    int N = g->N;
    int* dist = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) dist[i] = 0; // start from 0 for all

    for (int i = 0; i < N - 1; i++) {
        int updated = 0;
        for (int u = 0; u < N; u++) {
            for (Edge* e = g->adj[u]; e; e = e->next) {
                int v = e->to, w = e->weight;
                if (dist[u] != INF && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    updated = 1;
                }
            }
        }
        if (!updated) break;
    }

    // Check one more time for negative cycle
    for (int u = 0; u < N; u++) {
        for (Edge* e = g->adj[u]; e; e = e->next) {
            int v = e->to, w = e->weight;
            if (dist[u] != INF && dist[u] + w < dist[v]) {
                free(dist);
                return 1; // negative cycle found
            }
        }
    }
    free(dist);
    return 0; // no negative cycle
}

// Removes one edge from one negative cycle found using Bellman-Ford
void remove_one_negative_edge(Graph* g) {
    int N = g->N;
    int* dist = malloc(N * sizeof(int));
    int* pred = malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        dist[i] = 0;
        pred[i] = -1;
    }

    int cycle_start = -1;

    // Bellman-Ford to detect negative cycle
    for (int i = 0; i < N; i++) {
        cycle_start = -1;
        for (int u = 0; u < N; u++) {
            for (Edge* e = g->adj[u]; e; e = e->next) {
                int v = e->to;
                int w = e->weight;
                if (dist[u] != INF && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pred[v] = u;
                    cycle_start = v;
                }
            }
        }
        if (cycle_start != -1) break; // negative cycle detected
    }

    if (cycle_start == -1) {
        // No negative cycle
        free(dist);
        free(pred);
        return;
    }

    // Find cycle vertex by going predecessor N times
    int v = cycle_start;
    for (int i = 0; i < N; i++) {
        v = pred[v];
    }

    // Remove one edge from the cycle
    int u = v;
    do {
        int from = pred[u];
        Edge** prev = &g->adj[from];
        while (*prev) {
            if ((*prev)->to == u) {
                Edge* to_remove = *prev;
                *prev = (*prev)->next;
                free(to_remove);
                free(dist);
                free(pred);
                return;
            }
            prev = &(*prev)->next;
        }
        u = from;
    } while (u != v);

    free(dist);
    free(pred);
}

int main() {
    int num_nodes_options[] = {1000, 5000, 10000};
    float density_options[] = {0.3f, 0.5f, 0.7f};
    int k_values[] = {5, 10, 15, 20, 30, 50, 100, 200};

    for (int ni = 0; ni < 3; ni++) {
        int N = num_nodes_options[ni];
        for (int di = 0; di < 3; di++) {
            float dens = density_options[di];
            printf("\n--- Graph N=%d, density=%.2f ---\n", N, dens);
            Graph* g = create_graph(N);
            generate_graph(g, dens);
            make_strongly_connected(g);

            // Remove negative cycles by repeatedly removing one edge until none left
           // while (has_negative_cycle(g)) {
            //    remove_one_negative_edge(g);
            //}

            for (int ki = 0; ki < 8; ki++) {
                int k = k_values[ki];
                compute_and_report(g, k, 0);
                compute_and_report(g, k, 1);
            }
            // oslobađanje g je prepušteno radi sažetosti
        }
    }
    return 0;
}
