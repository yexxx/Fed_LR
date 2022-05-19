#include <time.h>
#include <unistd.h>

#include <fstream>
#include <random>

#include "ckks_and_helper.hpp"
#include "compute_enc.hpp"
#include "compute_n.hpp"

int main() {
    // init paras
    bool enc_flag = false;
    bool fed_flag = true;
    int n_iter = 10;
    // int samples=10000, n_iter = 100, segement_point = 7;
    double lr = 0.05, lambda = 10, bias = 0;
    cout << "enter para (one by one): enc_flag, n_iter\n";
    cin >> enc_flag >> n_iter;
    cout << endl;

    // load data
    string dataset_dir = "../dataset/default_credit_hetero_guest.csv";
    string dataset_dir_host = "../dataset/default_credit_hetero_host.csv";
    vector<double> y_whole;
    vector<vector<double>> dataset = ReadDatasetFromCSV(dataset_dir);
    for (vector<vector<double>>::iterator i = dataset.begin(); i != dataset.end(); ++i) {
        y_whole.push_back((*i)[0]);
        (*i).erase((*i).begin());
    }
    vector<double> y(y_whole.begin(), y_whole.begin() + 15000);
    vector<vector<double>> XA(dataset.begin(), dataset.begin() + 15000);
    vector<vector<double>> XBB = ReadDatasetFromCSV(dataset_dir_host, true);
    vector<vector<double>> XB(XBB.begin(), XBB.begin() + 15000);
    size_t size = XA.size();
    // vector<vector<double>> X(dataset.begin(), dataset.begin()+(int)(dataset.size()*0.75));
    // vector<vector<double>> X_test(dataset.begin()+(int)(dataset.size()*0.75),dataset.end());
    // vector<double> y(y_whole.begin(),y_whole.begin()+(int)(y_whole.size()*0.75));
    // vector<double> y_test(y_whole.begin()+(int)(y_whole.size()*0.75),y_whole.end());

    // // segement data
    // vector<vector<double>> XA, XB, XA_test, XB_test;
    for (vector<vector<double>>::iterator ix = XB.begin(); ix != XB.end(); ++ix) {
        // XA.push_back(vector<double>((*ix).end()-segement_point, (*ix).end()));
        // vector<double> temp_xb = (vector<double>((*ix).begin(), (*ix).end()-segement_point));
        (*ix).insert((*ix).begin(), 1);
        // XB.push_back(temp_xb);
    }
    // for(vector<vector<double>>::iterator ixt=X_test.begin(); ixt!=X_test.end(); ++ixt){
    //     XA_test.push_back(vector<double>((*ixt).end()-segement_point, (*ixt).end()));
    //     vector<double> temp_xbt = (vector<double>((*ixt).begin(), (*ixt).end()-segement_point));
    //     temp_xbt.insert(temp_xbt.begin(),1);
    //     XB_test.push_back(temp_xbt);
    // }
    // size_t size = XA.size();
    cout << "XA.shape:(" << size << "," << XA[0].size() << ")\tXB.shape:(" << size << "," << XB[0].size() << ")\n";

    // // clear data not use after
    // dataset.clear();
    // dataset_.clear();
    // X.clear();
    // X_test.clear();

    // start timing
    time_t start_time = time(0);

    // weights of client A, B
    vector<double> weightsA(XA[0].size(), 0.01), weightsB(XB[0].size(), 0.01);

    // C task 1: init CKKS and generate keys
    SEALContext context = SetupCKKS();
    if (enc_flag) {
        cout << "CKKS: " << context.parameter_error_message() << ":\n";
        print_parameters(context);
    }
    KeyGenerator keygen(context);
    SecretKey secrect_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    double scale = pow(2.0, 40);

    // start train
    for (int n_iter_ = 0; n_iter_ < n_iter; ++n_iter_) {
        // lr *= 0.7;
        cout << "\niter " << n_iter_ + 1 << ":\n";
        // A task 1: compute z_a, u_a, z_a_squre, in plaintext
        vector<double> z_a = compute_z_ab(XA, weightsA, size, bias);
        vector<double> u_a = compute_u_a(z_a, size);
        vector<double> z_a_squre = compute_z_a_squre(z_a, size);

        // B task 1: compte z_b, u_b, in plaintext
        vector<double> z_b = compute_z_ab(XB, weightsB, size, bias);
        vector<double> u_b = compute_u_b(y, z_b, size);
        double temp_bias = 0;
        for (size_t i = 0; i < z_b.size(); ++i) { temp_bias = temp_bias + z_b[i] - y[i] - bias; }
        bias = bias - temp_bias * lr / (double)z_b.size();
        if (enc_flag) {
            // A task1: enc u_a, z_a_squre
            Plaintext u_a_plaintext, z_a_squre_plaintext;
            Encode(context, u_a, scale, u_a_plaintext);
            Ciphertext encrypted_u_a = Encrypt(context, public_key, scale, u_a_plaintext);
            Encode(context, z_a_squre, scale, z_a_squre_plaintext);
            Ciphertext encrypted_z_a_squre = Encrypt(context, public_key, scale, z_a_squre_plaintext);

            // B task 1: enc u_b
            Plaintext u_b_plaintext;
            Encode(context, u_b, scale, u_b_plaintext);
            Ciphertext encrypted_u_b = Encrypt(context, public_key, scale, u_b_plaintext);

            // A task 2: compute ua, dJ_a, dJ_a+mask_a
            Ciphertext encrypted_ua = compute_encrypted_uab(context, encrypted_u_a, encrypted_u_b);
            vector<Ciphertext> encrypted_dJ_a =
                compute_encrypted_dJ_ab(context, relin_keys, XA, encrypted_ua, lambda, weightsA, scale);
            vector<vector<double>> mask_a(encrypted_dJ_a.size(), vector<double>(size));
            srand(time(0));
            for (size_t i = 0; i < mask_a.size(); ++i)
                for (size_t j = 0; j < mask_a[0].size(); ++j) mask_a[i][j] = (double)rand() / (double)RAND_MAX;
            vector<Ciphertext> masked_encrypted_dJ_a =
                compute_masked_encrypted_dJ_ab(context, encrypted_dJ_a, mask_a, scale);

            // B task 2: compute ub, dJ_b, dJ_b+mask_b, z, loss
            Ciphertext encrypted_ub = compute_encrypted_uab(context, encrypted_u_a, encrypted_u_b);
            vector<Ciphertext> encrypted_dJ_b =
                compute_encrypted_dJ_ab(context, relin_keys, XB, encrypted_ub, lambda, weightsB, scale);
            vector<vector<double>> mask_b(encrypted_dJ_b.size(), vector<double>(size));
            srand(time(0));
            for (size_t i = 0; i < mask_b.size(); ++i)
                for (size_t j = 0; j < mask_b[0].size(); ++j) mask_b[i][j] = (double)rand() / (double)RAND_MAX;
            vector<Ciphertext> masked_encrypted_dJ_b =
                compute_masked_encrypted_dJ_ab(context, encrypted_dJ_b, mask_b, scale);
            Ciphertext encrypted_z = compute_encrypted_z(context, relin_keys, encrypted_u_a, z_b, scale);
            Ciphertext encrypted_loss = compute_encrypted_loss(context, relin_keys, y, encrypted_z, encrypted_z_a_squre,
                                                               z_b, encrypted_u_a, scale);

            // C task 2: dec loss, dJ_a, dJ_b
            dec_compute_loss(context, secrect_key, encrypted_loss, size);
            vector<vector<double>> dJ_a = dec_dJ_ab(context, secrect_key, masked_encrypted_dJ_a, size);
            vector<vector<double>> dJ_b = dec_dJ_ab(context, secrect_key, masked_encrypted_dJ_b, size);

            // A task 3: revert dJ_a and update weightsA
            revert_dJ_ab_update_weightsAB(weightsA, mask_a, dJ_a, lr, size, "A");

            // B task 3: revert dJ_b and update weightsB
            revert_dJ_ab_update_weightsAB(weightsB, mask_b, dJ_b, lr, size, "B");
        } else {
            // A task 2: compute ua, dJ_a
            vector<double> ua = compute_u(u_a, u_b, size);
            vector<double> dJ_a;
            if (fed_flag)
                dJ_a = compute_dJ(XA, ua, weightsA, lambda, size);
            else
                dJ_a = compute_dJ(XA, u_a, weightsA, lambda, size);

            // B task 2: compute ub, dJ_b, z, loss
            vector<double> ub = compute_u(u_a, u_b, size);
            vector<double> dJ_b;
            if (fed_flag)
                dJ_b = compute_dJ(XB, ub, weightsB, lambda, size);
            else
                dJ_b = compute_dJ(XB, u_b, weightsB, lambda, size);
            vector<double> z = compute_z(u_a, z_b);
            double loss = compute_loss(z, y, z_a_squre, z_b, u_a);

            // C task 2: compute loss
            loss = loss / double(size) + log(2);
            cout << "************************************************loss: " << loss
                 << "************************************************" << endl;

            // A task 3: update weightsA
            update_weights(weightsA, dJ_a, lr, size, "A");

            // B task 3: update weightsB
            update_weights(weightsB, dJ_b, lr, size, "B");
        }
        compute_acc(weightsA, weightsB, XA, XB, y);
    }
    cout << "total time: " << difftime(time(0), start_time) << "s\n";
    return 0;
}