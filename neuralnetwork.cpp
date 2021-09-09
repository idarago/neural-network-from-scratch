#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <random>
#include <algorithm>

using namespace Eigen;

// Tensor product of matrices:
//  =============================================================
MatrixXf KroneckerProduct(const MatrixXf& A, const MatrixXf& B) {
    MatrixXf kronecker(A.rows()*B.rows(),A.cols()*B.cols());
    for (int i = 0; i<A.rows()*B.rows(); i++) {
        for (int j=0; j<A.cols()*B.cols(); j++) {
            kronecker(i,j) = A(i/B.rows(),j/B.cols())*B(i%B.rows(),j%B.cols());
        }
    }
    return kronecker;
}


// Initializes an mxn matrix where each coordinate is normally distributed (with mean 0 and std_dev 1)
//  =============================================================
MatrixXf RandNormalMatrix(int m, int n, float mean=0.0, float std=1.0) {
    MatrixXf mat(m,n);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution{mean,std};
    for (int i = 0; i < m; i++) {
        for (int j=0; j<n;j++) {
            mat(i,j) = distribution(generator);
        }
    }
    return mat;
}

// Loss functions:
//  =============================================================

// Calculation of Mean Squared Error function and its derivative:
//  =============================================================
float MeanSquaredError(const MatrixXf& y_pred, const MatrixXf& y_true){
    const int rows = y_pred.rows();
    // Calculate the mean squared error loss function
    // We set y = matrix where each column is the (true or predicted) value according to some input x
    // If (y1^,...,yn^) is my prediction and (y1,...,yn) is the true value, then the MSE
    // is calculated as MSE = 1/n ((y1-y1^)^2 + ... + (yn-yn^)^2)
    // This loss is summed over all columns to obtain the total loss from the sample.
    return (y_true-y_pred).colwise().squaredNorm().sum()/rows;
}

// Derivative of the Mean Squared Error function:
MatrixXf MSEgradient(const MatrixXf& values, const MatrixXf& y_true) {
    const int rows = y_true.rows();
    return (-2*(y_true - values))/rows;
}


// Argmax function
//  =============================================================
int argmax(const MatrixXf& v) {
    int index = 0;
    float maxval = v(0,0);
    for (int i=0; i < v.rows(); i++){ 
        if (v(i,0)>=maxval) {
            index = i;
            maxval=v(i,0);
        } 
    }
    return index;
}

// Implementing the softmax function and its derivative
//  =============================================================
MatrixXf softmax(const MatrixXf& x) {
    MatrixXf maxValues = x.colwise().maxCoeff();
    MatrixXf mat = MatrixXf::Zero(x.rows(),x.cols());
    MatrixXf result = MatrixXf::Zero(x.rows(),x.cols());
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j<x.cols(); j++) {
            mat(i,j) = exp(x(i,j) - maxValues(j));
        }
    }
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j<x.cols(); j++) {
            result(i,j) = mat(i,j) / (mat.col(j).sum());
        }
    }
    return result;
}

MatrixXf log(const MatrixXf& x) {
    MatrixXf result = MatrixXf::Zero(x.rows(),x.cols());
    for (int i=0; i < x.rows(); i++) {
        for (int j=0; j< x.cols(); j++) {
            result(i,j) = log(x(i,j));
        }
    }
    return result;
}


// Calculation of Categorical Cross-Entropy loss function and its derivative:
//  =============================================================

float CategoricalCrossEntropy(const MatrixXf& x, const MatrixXf& y_true) {
    const int rows = x.rows();
    MatrixXf y_pred = softmax(x);
    return (y_true.cwiseProduct(log(y_pred))).sum()/rows;
}
MatrixXf CCEgradient(const MatrixXf& x, const MatrixXf& y_true) {
    return (softmax(x) -  y_true).rowwise().mean();
}


// Activation functions:
//  =============================================================

//  Implementing ReLU and its derivative
//  =============================================================
float ReLU(const float& x) {
    return std::max(x,float(0.0));
}
// Vectorized version
MatrixXf ReLU(const MatrixXf& x) {
    MatrixXf result = MatrixXf::Zero(x.rows(),x.cols());
    for (int i = 0; i<x.rows(); i++) {
        for (int j = 0; j<x.cols(); j++) {
            result(i,j) = ReLU(x(i,j));
        }
    }
    return result;
}

// Derivative of ReLU function:
float dReLU_dx(const float& x) {
    return (x>0) ? 1.0 : 0.0;
}
// Vectorized version:
MatrixXf dReLU_dx(const MatrixXf& x) {
    MatrixXf result = MatrixXf::Zero(x.rows(),x.cols());
    for (int i=0; i<x.rows();i++) {
        for (int j=0; j<x.cols();j++) {
            result(i,j) = dReLU_dx(x(i,j));
        }
    }
    return result;
}

// Implementing Sigmoid function and its derivative
//  =============================================================
float sigmoid(const float& x) {
    return 1.0 / (1.0 + exp(-x));
}
// Vectorized version:
MatrixXf sigmoid(const MatrixXf& x) {
    MatrixXf result = MatrixXf::Zero(x.rows(),x.cols());
    for (int i=0; i<x.rows();i++) {
        for (int j=0;j<x.cols();j++) {
            result(i,j) = sigmoid(x(i,j));
        }
    }
    return result;
}

// Derivative of sigmoid function:
float dsigmoid_dx(const float& x) {
    // return sigmoid(x) * (1-sigmoid(x));
    return x * (1-x);
}
// Vectorized version:
MatrixXf dsigmoid_dx(const MatrixXf& x) {
    MatrixXf result = MatrixXf::Zero(x.rows(),x.cols());
    for (int i=0; i<x.rows();i++) {
        for (int j=0;j<x.cols();j++) {
            result(i,j) = dsigmoid_dx(x(i,j));
        }
    }
    return result;
}

//  =============================================================

class AbstractLayer {
public:
    // Constructor
    AbstractLayer(int input_size, int output_size) {
        _input_size = input_size;
        _output_size = output_size;
        _weights = RandNormalMatrix(output_size,input_size,0.01,0.1); // Randomly initialize weights
        _biases = VectorXf::Zero(output_size,1); // Initialize bias as 0
        _gradient_x = MatrixXf::Zero(input_size,1);
        _gradient_W = MatrixXf::Zero(output_size,input_size);
        _gradient_b = MatrixXf::Zero(output_size,1);
    }

    // Getter and setter methods
    void setWeights(const MatrixXf& weights) {
        _weights = weights;
    }
    MatrixXf getWeights() {
        return _weights;
    }
    void setBiases(const VectorXf& biases) {
        _biases = biases;
    }
    VectorXf getBiases() {
        return _biases;
    }
    void setInputs(const MatrixXf& inputs) {
        _inputs = inputs;
    }
    MatrixXf getInputs() {
        return _inputs;
    }
    MatrixXf getOutputs() {
        return _outputs;
    }
    MatrixXf getGradient() {
        return _gradient_x;
    }

    // Calculate output values from inputs, weights and biases
    // For each column vector x in inputs, we do (Wx+b) and then
    // apply the corresponding activation function depending on
    // the kind of layer
    virtual void forwardPass() = 0;

    virtual void backwardPass(const MatrixXf& D) = 0;

    // Stochastic gradient descent allows us to update the weights once
    // we've completed forward and backward pass
    void update(float learning_rate) {
        _weights -= learning_rate*_gradient_W;
        _biases -= learning_rate*_gradient_b;
    }

protected:
    int _input_size;
    int _output_size;
    MatrixXf _weights;
    VectorXf _biases;
    MatrixXf _inputs;
    MatrixXf _outputs;
    MatrixXf _gradient_W;
    MatrixXf _gradient_b;
    MatrixXf _gradient_x;
};

// Linear layer: the activation function is simply linear
//  =============================================================
class LinearLayer : public AbstractLayer {
public:
    LinearLayer(int input_size, int output_size) : AbstractLayer(input_size, output_size) {}
    void forwardPass() override {
        _outputs = (_weights*_inputs).colwise() + _biases;
    }
    void backwardPass(const MatrixXf& D) override {
        // Takes the average of the gradients when the input passed consists of many column vectors
        _gradient_x = _weights.transpose() * D;
        _gradient_b = D;
        _gradient_W = KroneckerProduct(_inputs.transpose().rowwise().mean(),_gradient_b);
    }
};

// ReLU layer: the activation is the ReLU function
//  =============================================================
class ReLULayer : public AbstractLayer {
public:
    ReLULayer(int input_size, int output_size) : AbstractLayer(input_size, output_size) {}
    void forwardPass() override {
        _outputs = ReLU((_weights*_inputs).colwise() + _biases);
    }
    void backwardPass(const MatrixXf& D) override {
        const MatrixXf z = (_weights*_inputs).colwise() + _biases;

        // Takes the average of the gradients when the input passed consists of many column vectors
        _gradient_x = _weights.transpose() * (dReLU_dx(z.col(0)).asDiagonal()) * D/_inputs.cols();
        for (int i = 1; i < _inputs.cols(); i++) {
            _gradient_x += _weights.transpose() * (dReLU_dx(z.col(i)).asDiagonal()) * D/_inputs.cols(); // _inputs here is a coincidence! it's actually sigma'(Wx+b) but sigma'(Wx+b) = sigma'(sigma(Wx+b)) magically
        }

        _gradient_b =  (dReLU_dx(z.col(0)).asDiagonal()) * D/_inputs.cols();
        for (int j = 1; j < _outputs.cols(); j++) {
            _gradient_b +=  (dReLU_dx(z.col(j)).asDiagonal()) * D/_inputs.cols();
        }

        _gradient_W = KroneckerProduct(_inputs.col(0).transpose() , dReLU_dx(z.col(0)).asDiagonal() * D)/_inputs.cols();
        for (int j = 1; j < _inputs.cols(); j++) {
            _gradient_W += KroneckerProduct(_inputs.col(j).transpose() , dReLU_dx(z.col(j)).asDiagonal() * D)/_inputs.cols();
        }
    }
};

// Sigmoid layer: the activation is the sigmoid function
//  =============================================================
class SigmoidLayer : public AbstractLayer {
public:
    SigmoidLayer(int input_size, int output_size) : AbstractLayer(input_size, output_size) {}
    void forwardPass() override {
        _outputs = sigmoid((_weights*_inputs).colwise() + _biases);
    }
    void backwardPass(const MatrixXf& D) override {
        const MatrixXf z = (_weights*_inputs).colwise() + _biases;
        // Takes the average of the gradients when the input passed consists of many column vectors
        _gradient_x = _weights.transpose() * (dsigmoid_dx(z.col(0)).asDiagonal()) * D/_inputs.cols();
        for (int i = 1; i < _inputs.cols(); i++) {
            _gradient_x += _weights.transpose() * (dsigmoid_dx(z.col(i)).asDiagonal()) * D/_inputs.cols(); // _inputs here is a coincidence! it's actually sigma'(Wx+b) but sigma'(Wx+b) = sigma'(sigma(Wx+b)) magically
        }

        _gradient_b =  (dsigmoid_dx(z.col(0)).asDiagonal()) * D/_inputs.cols();
        for (int j = 1; j < _outputs.cols(); j++) {
            _gradient_b +=  (dsigmoid_dx(z.col(j)).asDiagonal()) * D/_inputs.cols();
        }

        _gradient_W = KroneckerProduct(_inputs.col(0).transpose() , dsigmoid_dx(z.col(0)).asDiagonal() * D)/_inputs.cols();
        for (int j = 1; j < _inputs.cols(); j++) {
            _gradient_W += KroneckerProduct(_inputs.col(j).transpose() , dsigmoid_dx(z.col(j)).asDiagonal() * D)/_inputs.cols();
        }
    }
};


// Sequential layer: puts together our layers to form the model
//  =============================================================
class SequentialLayer {
public:
    SequentialLayer() {}
    std::vector<AbstractLayer*> getLayers() {
        return _layers;
    }
    // Method to add a layer to our neural network
    void add(AbstractLayer* layer) {
        _layers.push_back(layer);
    }
    // Setter and getters
    void setInputs(const MatrixXf& inputs) {
        _inputs = inputs;
    }
    void setLearningRate(const float& learning_rate) {
        _learning_rate = learning_rate;
    }
    void setBatchSize(const int& batch_size) {
        _batch_size = batch_size;
    }

    MatrixXf getPredictions() {
        return _predictions;
    } 

    // Pass the inputs to the first layer and propagate through the entire network
    void forwardPass() {
        _layers[0]->setInputs(_inputs);
        _layers[0]->forwardPass();
        for (std::size_t i = 1; i < _layers.size(); i++) {
            _layers[i]->setInputs(_layers[i-1]->getOutputs());
            _layers[i]->forwardPass();
        }
        _predictions = _layers[_layers.size()-1]->getOutputs();
    }
    // Do the backward pass starting with D (the differential of the loss)
    // at the last layer, going towards the first layer.
    void backwardPass(const MatrixXf& D) {
        _layers[_layers.size()-1]->backwardPass(D);
        _layers[_layers.size()-1]->update(_learning_rate);
        for (int i=_layers.size()-2; i>=0;i--) {
            _layers[i]->backwardPass(_layers[i+1]->getGradient());
            _layers[i]->update(_learning_rate);
        }
    }

private:
    std::vector<AbstractLayer*> _layers;
    float _learning_rate;
    int _batch_size;
    MatrixXf _inputs;
    MatrixXf _predictions;
};


// The learner class allows us to train our model
//  =============================================================
class Learner {
public:
    Learner(const SequentialLayer& model) {
        _model = model;
    }
    // Fit method trains our model
    std::vector<float> fit(const MatrixXf& x_train, const MatrixXf& y_train, int epochs, int batch_size=10, float learning_rate=0.001) {
        _model.setLearningRate(learning_rate);
        _model.setBatchSize(batch_size);
        std::vector<float> losses;
        float loss;
        int iterations;
        for (int epoch=0; epoch<epochs; epoch++) {
            std::cout << "Epoch: " << epoch+1 << std::endl;
            loss = 0.0;
            iterations = 0;
            while (iterations < x_train.cols()/batch_size) {
                std::vector<int> batch(x_train.cols()); // Random subset of size batch_size of samples
                std::iota(batch.begin(),batch.end(),0);
                shuffle(batch.begin(), batch.end(),std::default_random_engine());
                batch.resize(batch_size);
                MatrixXf x_submatrix = MatrixXf::Zero(x_train.rows(),batch_size);
                for (int i = 0; i < batch_size; i++) {
                    for (int j = 0; j < x_train.rows(); j++) {
                        x_submatrix(j,i) = x_train(j,batch[i]);
                    }
                } 
                MatrixXf y_submatrix = MatrixXf::Zero(y_train.rows(),batch_size);
                for (int i = 0; i < batch_size; i++) {
                    for (int j = 0; j < y_train.rows(); j++) {
                        y_submatrix(j,i) = y_train(j,batch[i]);
                    }
                }
                loss += fitBatch(x_submatrix,y_submatrix);
                losses.push_back(loss);
                iterations++;
            }
        }
        return losses;
    }

    MatrixXf Predictions(const MatrixXf& input) {
        _model.setInputs(input);
        _model.forwardPass();
        //return _model.getPredictions();
        
        // Softmax prediction -> Use this for MNIST
        return softmax(_model.getPredictions());
    }

private:
    SequentialLayer _model;

    float fitBatch(const MatrixXf& x_train, const MatrixXf& y_train) {
        _model.setInputs(x_train);
        _model.forwardPass();
        
        // Mean Squared Error
        //MatrixXf D = MSEgradient(_model.getPredictions(),y_train);
        
        // Categorical entropy + softmax -> Use this for MNIST
        MatrixXf D = CCEgradient(_model.getPredictions(),y_train);
        _model.backwardPass(D);
        _model.forwardPass();
        
        // Mean Squared Error:
        //return MeanSquaredError(_model.getPredictions(),y_train);
        
        // Categorical entropy + softmax -> Return this for MNIST
        return CategoricalCrossEntropy(_model.getPredictions(),y_train);
    }
};

void classificationToyCase(){
    const int samples = 100;
    MatrixXf x_train = RandNormalMatrix(1,samples);
    MatrixXf y_train = MatrixXf::Zero(2,samples);
    for (int i = 0; i < samples; i++) {
        if (x_train(0,i) > 0) y_train(0,i) = 1;
        else y_train(1,i) = 1;
    }

    ReLULayer layer(1,2);
    SequentialLayer model;
    model.add(&layer);

    Learner learner = Learner(model);
    std::vector<float> losses = learner.fit(x_train, y_train, 1000, 100, 0.01); // epochs = 10, batch_size = 60000

    MatrixXf x_test = RandNormalMatrix(1,10);
    for (int i = 0; i < 10; i++) { 
        std::cout << "Actual value: " << x_test.col(i) << std::endl;
        std::cout << "Prediction: " << learner.Predictions(x_test.col(i)) << std::endl;
        std::cout << "Predicted value: " << argmax(learner.Predictions(x_test.col(i))) << std::endl;
    }
}

int main() {
    // Loading the MNIST data from the formatted .txt files
    std::cout << "Loading MNIST data..." << std::endl;
    FILE* traindata;
    FILE* testdata;
    FILE* trainlabel;
    FILE* testlabel;
    traindata = fopen("mnist_train.txt", "r");
    testdata = fopen("mnist_test.txt", "r");
    trainlabel = fopen("mnist_train_labels.txt", "r");
    testlabel = fopen("mnist_test_labels.txt", "r");

    int train_samples;
    fscanf(traindata, "%d", &train_samples);
    fscanf(trainlabel, "%d", &train_samples);
    int onehot;
    MatrixXf x_train = MatrixXf::Zero(784,train_samples);
    MatrixXf y_train = MatrixXf::Zero(10,train_samples);
    for (int i = 0; i < train_samples; i++) {
        for (int j = 0; j < 784; j++) {
            fscanf(traindata, "%f", &x_train(j,i));
        }
        fscanf(trainlabel, "%d", &onehot);
        y_train(onehot,i)=1;
    }

    int test_samples;
    fscanf(testdata, "%d", &test_samples);
    fscanf(testlabel, "%d", &test_samples);
    MatrixXf x_test = MatrixXf::Zero(784,test_samples);
    MatrixXf y_test = MatrixXf::Zero(10,test_samples);
    for (int i = 0; i < test_samples; i++) {
        for (int j = 0; j < 784; j++) {
            fscanf(testdata, "%f", &x_test(j,i));
        }
        fscanf(testlabel, "%d", &onehot);
        y_test(onehot,i)=1;
    }

    // Add layers to the model
    std::cout << "Initializing the model..." << std::endl;
    ReLULayer layer(784,64);
    ReLULayer layer2(64,10);
    SequentialLayer model;
    model.add(&layer);
    model.add(&layer2);

    // Train the model
    std::cout << "Training the model" << std::endl;
    Learner learner = Learner(model);
    std::vector<float> losses = learner.fit(x_train, y_train, 5, 60000, 0.01); // epochs = 10, batch_size = 60000

    // Evaluation metrics
    int TEST_SIZE = test_samples;
    int count=0;
    for (int i = 0; i<TEST_SIZE;i++){
        std::cout << "Prediction: " << learner.Predictions(x_test.col(i)) << std::endl;
        std::cout << "Predicted number " << argmax(learner.Predictions(x_test.col(i))) << std::endl;
        std::cout << " " << std::endl;
        std::cout << "Actual number: " << argmax(y_test.col(i)) << std::endl;
        std::cout << "Just in case: " << (y_test.col(i)) << std::endl;
        std::cout << "--------" << std::endl;
        if (argmax(learner.Predictions(x_test.col(i))) == argmax(y_test.col(i))) count++;
    }
    std::cout << "Acc " << 100.0*count/TEST_SIZE << "%";

    return 0;
}
