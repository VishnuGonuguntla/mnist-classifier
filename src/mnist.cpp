#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <chrono>
using namespace std;
using Eigen::MatrixXf, Eigen::MatrixXd, Eigen::MatrixXi;
using Eigen::all, Eigen::seq;
int source = 60000, test=10000, n = 784, k = 10, q = 1;
int BigEndiantoEndian (int i) {
    unsigned char c1, c2, c3, c4;
    c1=i&255;
    c2=(i>>8)&255;
    c3=(i>>16)&255;
    c4=(i>>24)&255;
    return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}
map<string,string> configsrc(string config_src) {
    map<string,string> config_data;
    ifstream file(config_src);
    string line;
    if (!file.is_open()) {
        cout << "Error opening file " << config_src << endl;
        return config_data;
    }
    while (getline(file,line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t pos = line.find(" = ");
        int i = 0;
        if (pos!=string::npos) {
            string key = line.substr(0, pos);
            string value = line.substr(pos+3);
            config_data[key] = value;
        }
    }
    return config_data;
}
void shuffle_data (MatrixXd& image_data, MatrixXd& label_data, vector<int>& shuffle_index_data, MatrixXd& shuffled_image, MatrixXd& shuffled_label){
    random_device rd;
    mt19937 gen(rd());
    shuffle(shuffle_index_data.begin(),shuffle_index_data.end(), gen);
    for (int i = 0; i < source ; i++) {
        for (int j_image = 0; j_image < n ; j_image++) {
            shuffled_image(i,j_image) = image_data(shuffle_index_data[i],j_image);
        }
        for (int j_label = 0; j_label < k ; j_label++) {
            shuffled_label(i,j_label) = label_data(shuffle_index_data[i],j_label);
        }
    }
}
void src_data_reading(MatrixXd& image_data, MatrixXd& label_data, string train_images, string train_labels,string type,MatrixXd& test_true) {
    // Image Data Reading:
    ifstream image_source(train_images,ios::binary);
    if (image_source.is_open()){
        int magic_number=0;
        int number_of_images=0;
        int number_of_rows=0;
        int number_of_columns=0;
        image_source.read((char*)&magic_number,sizeof(magic_number));
        magic_number= BigEndiantoEndian(magic_number);
        image_source.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= BigEndiantoEndian(number_of_images);
        image_source.read((char*)&number_of_rows,sizeof(number_of_rows));
        number_of_rows= BigEndiantoEndian(number_of_rows);
        image_source.read((char*)&number_of_columns,sizeof(number_of_columns));
        number_of_columns= BigEndiantoEndian(number_of_columns);
        // Pixel Data Allocation
        for(int i=0;i<number_of_images;++i){
            for(int r=0;r<number_of_rows;++r){
                for(int c=0;c<number_of_columns;++c){
                    uint8_t temp=0;
                    image_source.read((char*)&temp, sizeof(temp));
                    image_data(i,(number_of_rows*r)+c)= temp/255.0f;
                    if (temp/255.0f == 0.313726) image_data(i,(number_of_rows*r)+c)=0.313725;
                }
            }
        }
    }
    else {
        cout<<"Error opening file"<<endl;
    }
    // Label Data Reading:
    ifstream label_source (train_labels,ios::binary);
    if (label_source.is_open()){
        int magic_number=0;
        int number_of_images=0;
        label_source.read((char*)&magic_number,sizeof(magic_number));
        magic_number= BigEndiantoEndian(magic_number);
        label_source.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= BigEndiantoEndian(number_of_images);
        for(int i=0;i<number_of_images;++i) {
            uint8_t temp=0;
            label_source.read((char*)&temp,sizeof(temp));
            label_data(i,temp)= 1.0f;
            if (type == "test") test_true(i,0) = temp;
        }
    } else {
        cout<<"Unable to open file" << endl;
    }
    label_source.close();
}
void testing (MatrixXd& weighted_matrix_1, MatrixXd& weighted_matrix_2, MatrixXd& test_image_data,MatrixXd& test_label_data,MatrixXd& test_true_label_data,int batch, string output,int hidden_size) {
    cout << "Testing Phase ";
    auto start= chrono::steady_clock::now();
    MatrixXd input_layer = MatrixXd::Ones(batch,n+1);
    MatrixXd updated_activation_layer_1 = MatrixXd::Ones(batch,hidden_size+1);
    MatrixXd hidden_layer_1, activation_layer_1;
    MatrixXd hidden_layer_2, activation_layer_2(batch,k);
    MatrixXd activation_layer_2_sum;
    ofstream file(output);
    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        exit(0);
    }
    int colIndex = 0;
    double maxValue = 0;
    for (int iter = 0; iter < test_image_data.rows()/batch; iter++) {
        file << "Current batch: " << iter<<endl;
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < test_image_data.cols(); j++) {
                input_layer(i,j) = test_image_data(iter*batch + i,j);
            }
        }
        hidden_layer_1 = input_layer*weighted_matrix_1;
        activation_layer_1 = hidden_layer_1.cwiseMax(0);
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < hidden_size; j++) {
                updated_activation_layer_1(i,j) = activation_layer_1(i,j);
            }
        } // Adds a row with ones at the end
        hidden_layer_2 = updated_activation_layer_1*weighted_matrix_2;
        hidden_layer_2 = (hidden_layer_2.array() - hidden_layer_2.maxCoeff()).array().exp();
        activation_layer_2_sum = hidden_layer_2.rowwise().sum();
        for (int i = 0; i < batch; i++) {
            activation_layer_2.row(i) = hidden_layer_2.row(i).array()/activation_layer_2_sum(i,0);
        }
        for (int i = 0; i < batch; i++){
            int actual = test_true_label_data(iter*batch+i,0);
            activation_layer_2.row(i).maxCoeff(&colIndex);
            file <<"- image "<<i<<" predicted: "<<colIndex<<". Label="<<actual<<endl;
        }
    }
    file.close();
    auto end = chrono::steady_clock::now();
    cout<<std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << endl;
}
int main(int argc, const char* argv[]) {
    string config_src = argv[1];
    map<string,string> config_file = configsrc(config_src);
    int num_epochs =  stoi(config_file["num_epochs"]);
    int batch = stoi(config_file["batch_size"]);
    int m = stoi(config_file["hidden_size"]);
    int eta = stoi(config_file["learning_rate"]);
    string rel_path_train_images = config_file["rel_path_train_images"];
    string rel_path_train_labels = config_file["rel_path_train_labels"];
    string rel_path_test_images = config_file["rel_path_test_images"];
    string rel_path_test_labels = config_file["rel_path_test_labels"];
    string rel_path_log_file = config_file["rel_path_log_file"];
    // Network Data:
    MatrixXd input_layer = MatrixXd::Ones(batch,n+1);
    MatrixXd weighted_matrix_1 = MatrixXd::Random(n+1,m),weighted_matrix_2= MatrixXd::Random(m+1,k);
    MatrixXd hidden_layer_1(batch,m),activation_layer_1(batch,m);
    MatrixXd hidden_layer_2(batch,k),activation_layer_2(batch,k); // FC2
    MatrixXd updated_activation_layer_1 = MatrixXd::Ones(batch,m+1);
    // Back Propagation Matrix:
    MatrixXd gradient_1(n+1,m),gradient_2(m+1,k);
    MatrixXd hidden_layer_1_back(batch,m+1),hidden_layer_2_back(batch,k);
    // Source Data:
    MatrixXd src_image_data(source,n),src_label_data(source,k);
    MatrixXd test_image_data(test,n), test_label_data(test,k);
    MatrixXd shuffled_image(source,n),shuffled_label(source,k);
    MatrixXd test_true_label_data(test,q);
    auto start= chrono::steady_clock::now();
    src_data_reading(src_image_data,src_label_data,rel_path_train_images,rel_path_train_labels,"train",test_true_label_data);
    src_data_reading(test_image_data,test_label_data,rel_path_test_images,rel_path_test_labels,"test",test_true_label_data);
    auto end = chrono::steady_clock::now();
    cout<< "Dataloadingdone " << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << endl;
    vector<int> shuffle_index_data(source,0);
    for (int i = 0; i < source; i++) shuffle_index_data[i] = i;
    MatrixXd batch_image_data(batch,n);
    MatrixXd batch_label_data(batch,k);
    start= chrono::steady_clock::now();
    cout << "Training ..." ;
    for (int epoch =0; epoch <num_epochs; epoch++) {
        shuffle_data(src_image_data,src_label_data,shuffle_index_data,shuffled_image,shuffled_label);
        for (int iter = 0; iter < test_image_data.rows()/batch; iter++) {
            for (int i = 0; i < batch; i++) {
                for (int j_image = 0; j_image < n; j_image++) {
                    input_layer(i,j_image) = shuffled_image(iter*batch+i,j_image);
                }
                for (int j_label = 0; j_label < k; j_label++) {
                    batch_label_data(i,j_label) = shuffled_label(iter*batch+i,j_label);
                } // Matrix with batched data for this specific iteration.
            } // Batching
            // Forward Propagation:
            hidden_layer_1 = input_layer*weighted_matrix_1; //FC1
            activation_layer_1 = hidden_layer_1.cwiseMax(0); // ReLU
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < m; j++) {
                    updated_activation_layer_1(i,j) = activation_layer_1(i,j);
                }
            }
            hidden_layer_2 = updated_activation_layer_1*weighted_matrix_2; //FC2
            hidden_layer_2 = (hidden_layer_2.array() - hidden_layer_2.maxCoeff()).array().exp();
            MatrixXd softmax_activation_layer_sum = hidden_layer_2.rowwise().sum();
            for (int i = 0; i < batch; i++) {
                activation_layer_2.row(i) = hidden_layer_2.row(i)/softmax_activation_layer_sum(i,0);
            }
            // back_Propagation:
            hidden_layer_2_back = activation_layer_2-batch_label_data; //dz2 100,10
            MatrixXd hidden_layer_2_bias = hidden_layer_2_back.colwise().sum()/batch; // 1,10
            MatrixXd hidden_layer_2_weight = activation_layer_1.transpose()*hidden_layer_2_back/batch; // (10,100)x(100,500) = 10,500
            gradient_2 <<  hidden_layer_2_weight,hidden_layer_2_bias ;//(m+1,k)
            MatrixXd del_ReLU = (hidden_layer_1.array()>0).cast<double>(); // (batch,m)
            hidden_layer_1_back = (hidden_layer_2_back*weighted_matrix_2(seq(0,weighted_matrix_2.rows()-2),all).transpose()).array()*del_ReLU.array(); //(batch,m)
            MatrixXd hidden_layer_1_bias = hidden_layer_1_back.colwise().sum()/batch; // (1,m)
            MatrixXd hidden_layer_1_weight = (input_layer(all,seq(0,input_layer.cols()-2)).transpose()*hidden_layer_1_back)/batch;
            gradient_1 << hidden_layer_1_weight,hidden_layer_1_bias;
            weighted_matrix_2 = weighted_matrix_2 - eta*gradient_2;
            weighted_matrix_1 = weighted_matrix_1 - eta*gradient_1;
        }
    }
    end = chrono::steady_clock::now();
    cout<< std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << endl;
    testing(weighted_matrix_1,weighted_matrix_2,test_image_data,test_label_data,test_true_label_data,batch, rel_path_log_file,m);
    return 0;
}