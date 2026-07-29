//
#include <map>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;

Vector as_1d(const Matrix& m){
    return m.rows() == 1 ? m.rows() : m.col(0);
}


enum FeatureType {
    CONTINUOUS, BINARY, CATEGORICAL
};
std::vector<FeatureType> infer_featurePtype(const Matrix& X){
	std::vector<FeatureType> types(X.cols(), CONTINUOUS);
    return types;
}


void train_test_split(const Matrix& X,
	                  const Vector& Y,
	                  const VectorXi& W,
	                  double test_size,
	                  int seed,
	                  Matrix& X_train,
	                  Matrix& X_test,
	                  Vector& Y_train,
	                  Vector& Y_test,
	                  VectorXi& W_train,
	                  VectorXi& W_test){
    int n = X.rows();
    int n_test = static_cast<int>(n * test_size);
    std::mt19937 rng(seed);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);//np.arange(idx[0], idx[-1])
    std::shuffle(idx.begin(), idx.end(), rng);
    X_train.resize(n - n_test, X.cols());
    X_test.resize(n_test, X.cols());
    Y_train.resize(n - n_test);
    Y_test.resize(n_test);
    W_train.resize(n - n_test);
    W_test.resize(n_test);
    ////train_test_split:
    for(int i = 0; i < n - n_test; ++i){
    	X_train.row(i) = X.row(idx[i]);
    	Y_train(i) = Y(idx[i]);
    	W_train(i) = W(idx[i]);
    }
    for(int i = 0; i < n_test; ++i){
    	int orig = idx[n - n_test + i];
    	X_test.row(i) = X.row(orig);
    	Y_test(i) = Y(orig);
    	W_test(i) = W(orig);
    }
}





double po_risk(const Vector& tau,
	           const Vector& Y,
	           const VectorXi& W,
	           const Vector& mu0,
	           const Vector& mu1,
	           const Vector& pi,
	           double clip = 0.005){
    double risk = 0.0;
    int n = tau.size();
    for(int i = 0; i < n; ++i){
    	pi(i) > clip ? pi(i) : clip;
    	pi(i) < 1.0 - clip ? pi(i) : (1.0 - clip);
    	double pred = mu0(i) + tau(i) * (W(i) - pi(i))/(pi(i) * (1 - pi));
    	double residual = Y(i) - pred;
    	risk += residual * residual;
    }
    return risk/n;
}




Vector permuCATE_vimp(
    const Matrix& X,
    const Vector& Y,
    const VectorXi& W,
    const ModelRegistry& model_registry,
    const std::string& model_m = "rf_regressor",
    const std::string& model_e = "logistic_classifier",
    const std::string& model_tau = "rf_regressor",
    const std::string& model_nu = "rf_regressor",
    const std::string& model_nu_bi = "logistic_classifier",
    const std::string& model_nu_multi = "multinomial_classifier",
    int n_perm = 100,
    double test_size = 0.5,
    int seed = 0,
    double clip_e = 0.01,
    bool normalize = false
){
    Matrix X_train, X_test;
    Vector Y_train, Y_test;
    VectorXi W_train, W_test;
    train_test_split(X, Y, W, test_size,seed,
        X_train, X_test, Y_train, Y_test,
        W_train, W_test)
    NuisanceOutput fitted = permuCATE_fit_nuisance(
        X_train, Y_train, W_train,
        model_registry, model_m
    )
}




























































































