#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include "seal/seal.h"
using namespace seal;
using namespace std;

SEALContext SetupCKKS(){
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 60}));
    return SEALContext(parms);
}

Ciphertext Encrypt(SEALContext &context, PublicKey &public_key, double &scale, Plaintext &plaintext){
    Encryptor encryptor(context, public_key);
    Ciphertext ciphertext;
    encryptor.encrypt(plaintext, ciphertext);
    return ciphertext;
}

Plaintext Decrypt(SEALContext &context, SecretKey &secret_key, Ciphertext &ciphertext){
    Decryptor decryptor(context, secret_key);
    Plaintext plaintext;
    decryptor.decrypt(ciphertext, plaintext);
    return plaintext;
}

void Encode(CKKSEncoder &encoder, vector<double> &input, double &scale, Plaintext &output){
    encoder.encode(input, scale, output);
}

void Decode(CKKSEncoder &encoder, Plaintext &input, vector<double> &output){
    encoder.decode(input, output);
}

void Encode(CKKSEncoder &encoder, double input, double &scale, Plaintext &output){
    encoder.encode(input, scale, output);
}

void Encode(SEALContext &context, vector<double> &input, double &scale, Plaintext &output){
    CKKSEncoder encoder(context);
    Encode(encoder, input, scale, output);
}

void Decode(SEALContext &context, Plaintext &input, vector<double> &output){
    CKKSEncoder encoder(context);
    Decode(encoder, input, output);
}

void Encode(SEALContext &context, double input, double &scale, Plaintext &output){
    CKKSEncoder encoder(context);
    Encode(encoder, input, scale, output);
}

inline void print_parameters(const seal::SEALContext &context){
    auto &context_data = *context.key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme()){
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    default:
        throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++){
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv){
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}

vector<vector<double>> ReadDatasetFromCSV(string filename, bool delete_id_flag=0)
{
    fstream fin;
    fin.open(filename, ios::in);

    vector<double> row;
    vector<vector<double>> dataset;
    string line, word;
    double value;

    getline(fin, line);
    while (fin.good())
    {
        row.clear();
        getline(fin, line);
        stringstream ssline(line);
        while (getline(ssline, word, ','))
        {
            stringstream ssword(word);
            ssword >> value;
            row.push_back(value);
        }

        dataset.push_back(row);
    }
    fin.close();
    // delete id
    if(delete_id_flag) for(size_t i=0; i<dataset.size(); i++) if(dataset[i].size()>0) dataset[i].erase(dataset[i].begin());
    return dataset;
}

