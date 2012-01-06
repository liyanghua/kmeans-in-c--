#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>

#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "util.hpp"
#include "data_loader.hpp"


using namespace std;
using namespace boost::numeric::ublas;


bool debug = false;

void generate_k_random_integers(set<int>& s, int upper_bound, int k) {
    boost::mt19937 rng;   
    boost::uniform_int<> g(1, upper_bound);
    for (size_t i=0; i<(size_t)k; ++i) {
        int x = g(rng) - 1;
        s.insert(x);
    }
}

void dump_clusters(const map<int, std::vector<int> >& clusters) {
    map<int, std::vector<int> >::const_iterator it(clusters.begin());
    for(; it != clusters.end(); ++it) {
        const std::vector<int>& v = it->second;
        int i = it->first;
        cout << "the " << i << "th cluster has " << v.size() << " points..." << endl;
        for (size_t x=0; x<v.size(); ++x) {
            cout << v[x] << " ";
        }
        cout << endl;
    }
    
}

void dump_vector(const std::vector<int>& v) {
    for (size_t i=0; i<v.size(); ++i) {
        cout << v[i] << "";
    }
    cout << endl;
}

void new_mean(boost::numeric::ublas::vector<double>& mean, 
        const std::vector<int>& cluster, // only store the index in the matrix
        boost::numeric::ublas::matrix<double>& x)
{
    if (debug) {
        cout << "compute the new mean for points list: " << endl;
        dump_vector(cluster);
    }

    // for each dim
    for (size_t i=0; i<mean.size(); ++i) {
        double m = 0;
        for (size_t j=0; j<cluster.size(); ++j) {
            matrix_row< matrix<double> > m_r(x, cluster[j]);
            m += m_r(i);
            assert(x(cluster[j],i) == m_r(i));
        }
        m /= cluster.size();
        mean(i) = m;
    }
    if (debug) {
        cout << "-----new mean----" << endl;
        cout << mean << endl;
    }
}


void get_new_means(const map<int, std::vector<int> >& clusters, 
        boost::numeric::ublas::matrix<double>& x,
        std::vector<boost::numeric::ublas::vector<double> >& cur_means,
        std::map<int, int>& mean_index_2_cluster_index
        )
{
    map<int, std::vector<int> >::const_iterator it(clusters.begin());
    // for each cluster, compute the mean point
    int mean_index = 0;
    for(; it != clusters.end(); ++it) {
        const std::vector<int>& v = it->second;
        if (v.size() <= 0) continue;

        boost::numeric::ublas::vector<double> mean(x.size2());
        new_mean(mean, v, x);
        cur_means.push_back(mean);

        int cluster_index = it->first;
        mean_index_2_cluster_index[cluster_index] = mean_index;
        mean_index += 1;
    }

}

double compute_target(const std::vector<boost::numeric::ublas::vector<double> >& means,
                    boost::numeric::ublas::matrix<double>& x,
                    map<int, std::vector<int> > clusters,
                    map<int, int> mean_index_2_cluster_index
                    )
{
    map<int, std::vector<int> >::iterator it(clusters.begin());
    double sum = 0;
    for (; it != clusters.end(); ++it) { // for each cluster
        double cluster_sum = 0;
        int cluster_index = it->first;
        std::vector<int>& data = it->second;
        // get the mean index according the cluster index
        int mean_index = mean_index_2_cluster_index[cluster_index];
        // get the mean
        boost::numeric::ublas::vector<double> mean(means[mean_index]);

        // for each point in the cluster
        for (size_t i=0; i<data.size(); ++i) {
            // get the real data
            matrix_row< matrix<double> > m(x, data[i]);
            // compute the norm between the data and the mean
            double partial_norm = norm(mean,m);
            if (debug)
                cout << "partial_norm: " << partial_norm << endl;
            cluster_sum += partial_norm;
        }

        sum += cluster_sum;
    }

    return sum;
}

void kmeans_method( boost::numeric::ublas::matrix<double>& x, const int k, const int r) {
    double target = 99999999;

    map<int, std::vector<int> > clusters;
    map<int, int> mean_index_2_cluster_index;

    std::vector<boost::numeric::ublas::vector<double> > means;
    means.reserve(k);

    // step 1: randomly generated k points, each point becomes the mean of the clustering
    set<int> s;
    generate_k_random_integers(s, r, k);

    // initial
    for (set<int>::iterator it=s.begin(); it != s.end(); ++it) {
        matrix_row< matrix<double> > m(x, *it);
        means.push_back(m);
        if (debug) {
            cout << *it << " point has been seletectd.." << endl; 
            cout << m << endl;
        }
    }

    // step 2: for each point, compute the distances from the mean points and assign the point to the nearest mean point
    size_t rows = x.size1();

    double e = 0.0001;
    long iter_num = 100;
    int start = 0;
    while (start++ < iter_num) {
        cout  << start << "th iteration,";
        for (size_t i=0; i<rows; ++i) {
            double min_dist = 9999; // set it to a big one
            size_t index = -1; 
            matrix_row< matrix<double> > m2(x, i);

            for (size_t j=0; j<means.size(); ++j) {
                double cur_dist = norm(means[j], m2);
                if (debug)
                    cout << "dist: " << cur_dist << endl;
                if (cur_dist < min_dist) {
                    min_dist = cur_dist;
                    index = j;
                }
            }
            clusters[index].push_back(i);
            if (debug)
                cout << "the " << i << "th point is assigned to the " << index << "th cluster." << endl;
        }
        if (debug)
            dump_clusters(clusters);
        
        means.clear();

        // step 3: compute the new means, repeated step 2 util the means of the clutering do not change.
        get_new_means(clusters, x, means, mean_index_2_cluster_index);

        // step 4: compare the two means, if they does not changed, stop
        double cur_target = compute_target(means, x, clusters, mean_index_2_cluster_index);
        cout << " target=" << cur_target << endl;


        mean_index_2_cluster_index.clear();
        if (abs(cur_target - target) < e) break;
        target = cur_target;
        clusters.clear();
    }

    cout << " The final clustering info: " << endl;
    dump_clusters(clusters);

}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " data_file" << endl;
        return -1;
    }

    const int record_num = 270;
    const int dim_num = 13;
    const int cluster_num = 5;

    boost::numeric::ublas::vector<double> y(record_num);
    boost::numeric::ublas::matrix<double> x(record_num, dim_num);
    SimpleDataLoader loader(record_num, dim_num);
    loader.load_file(argv[1], y, x);

    kmeans_method(x, cluster_num, record_num);

    return 0;
}
