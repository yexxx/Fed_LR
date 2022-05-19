#include <vector>
#include "seal/seal.h"
using namespace std;
using namespace seal;

// encrypted_uab = encrypted_u_a+encrypted_u_b
Ciphertext compute_encrypted_uab(SEALContext context, Ciphertext u_a, Ciphertext u_b){
    Evaluator evaluator(context);
    Ciphertext result;
    evaluator.add(u_a,u_b,result);
    return result;
}

// encrypted_dJ_a = XA.T.dot(encrypted_u)+lambda*weightsA -> n*1 vector
vector<Ciphertext> compute_encrypted_dJ_ab(SEALContext context, RelinKeys relin_keys, vector<vector<double>> XA, Ciphertext encrypted_u, double lambda, vector<double> weightsA, double scale){
    Evaluator evaluator(context);
    vector<Ciphertext> result(weightsA.size());

    vector<vector<double>> XT(XA[0].size(),vector<double>(XA.size()));
#pragma omp parallel for
    for(size_t i=0; i<XT.size(); ++i){
#pragma omp parallel for
        for(size_t j=0; j<XT[0].size(); j++){
            XT[i][j] = XA[j][i];
        }
    }
    
#pragma omp parallel for
    for(size_t i=0; i<XT.size(); ++i){
        Plaintext XT_i_plaintext, weightsA_i_plaintext;
        Encode(context,XT[i],scale,XT_i_plaintext);
        Encode(context,lambda*weightsA[i]/XT[0].size(),scale,weightsA_i_plaintext);
        evaluator.multiply_plain(encrypted_u,XT_i_plaintext,result[i]);
        evaluator.relinearize_inplace(result[i],relin_keys);
        evaluator.rescale_to_next_inplace(result[i]);
        result[i].scale() =  scale;
        evaluator.mod_switch_to_inplace(weightsA_i_plaintext,result[i].parms_id());
        evaluator.add_plain_inplace(result[i],weightsA_i_plaintext);
    }
    return result;
}

// compute encrypted_dJ_ab+mask_ab
vector<Ciphertext> compute_masked_encrypted_dJ_ab(SEALContext context, vector<Ciphertext> encrypted_dJ_ab, vector<vector<double>> mask_ab, double scale){
    Evaluator evaluator(context);
    vector<Ciphertext> result(encrypted_dJ_ab.size());
#pragma omp parallel for
    for(size_t i=0; i<encrypted_dJ_ab.size(); ++i){
        Plaintext plain_mask_ab;
        Encode(context,mask_ab[i],scale,plain_mask_ab);
        evaluator.mod_switch_to_inplace(plain_mask_ab,encrypted_dJ_ab[i].parms_id());
        evaluator.add_plain(encrypted_dJ_ab[i],plain_mask_ab,result[i]);
    }
    return result;
}

// compute encrypted_z = 4*encpypted_u_a+z_b
Ciphertext compute_encrypted_z(SEALContext context,RelinKeys relin_keys, Ciphertext encrypted_u_a, vector<double> z_b, double scale){
    Evaluator evaluator(context);
    Plaintext plain_4, plain_z_b;
    Encode(context,4.0,scale,plain_4);
    Ciphertext result;
    Encode(context,z_b,scale,plain_z_b);
    evaluator.multiply_plain(encrypted_u_a,plain_4,result);
    evaluator.relinearize_inplace(result,relin_keys);
    evaluator.rescale_to_next_inplace(result);
    result.scale() =  scale;
    evaluator.mod_switch_to_inplace(plain_z_b,result.parms_id());
    evaluator.add_plain_inplace(result,plain_z_b);
    return result;
}

// compute_encrypted_loss = sum((0.5-y)*encrypted_z+0.125*encrypted_z_a_squre+0.125*z_b*(encrypted_z+4*encrypted_u_a))
Ciphertext compute_encrypted_loss(
    SEALContext context, RelinKeys relin_keys, vector<double> y, Ciphertext encrypted_z, 
    Ciphertext encrypted_z_a_squre, vector<double> z_b, Ciphertext encrypted_u_a, double scale){
    Evaluator evaluator(context);
    Ciphertext result;

    // result = (0.5-y)*encrypted_z
    Plaintext plain_zp5_sub_y;
    vector<double> zp5_sub_y(y.size());
#pragma omp parallel for
    for(size_t i=0; i<zp5_sub_y.size(); ++i) zp5_sub_y[i]=0.5-y[i];
    Encode(context,zp5_sub_y,scale,plain_zp5_sub_y);
    evaluator.mod_switch_to_inplace(plain_zp5_sub_y,encrypted_z.parms_id());
    evaluator.multiply_plain(encrypted_z,plain_zp5_sub_y,result);
    evaluator.relinearize_inplace(result,relin_keys);
    evaluator.rescale_to_next_inplace(result);
    result.scale() = scale;
    
    // result = result+0.125*encrypted_z_a_squre
    Plaintext plain_0125;
    Encode(context,0.125,scale,plain_0125);
    Ciphertext temp0;
    evaluator.multiply_plain(encrypted_z_a_squre,plain_0125,temp0);
    evaluator.relinearize_inplace(temp0,relin_keys);
    evaluator.rescale_to_next_inplace(temp0);
    temp0.scale() = scale;
    evaluator.mod_switch_to_inplace(temp0,result.parms_id());
    evaluator.add_inplace(result,temp0);

    // result = result+0.125*z_b*(encrypted_z+4*encrypted_u_a)
    Plaintext plain_125_mul_z_b, plain_4;
    vector<double> o25_mul_z_b(z_b.size());
#pragma omp parallel for
    for(size_t i=0; i<z_b.size(); ++i) o25_mul_z_b[i] = 0.125*z_b[i];
    Encode(context,o25_mul_z_b,scale,plain_125_mul_z_b);
    Encode(context,4,scale,plain_4);
    Ciphertext temp1;
    evaluator.multiply_plain(encrypted_u_a,plain_4,temp1);
    evaluator.relinearize_inplace(temp1,relin_keys);
    evaluator.rescale_to_next_inplace(temp1);
    temp1.scale() =  scale;
    evaluator.mod_switch_to_inplace(temp1,encrypted_z.parms_id());
    evaluator.add_inplace(temp1,encrypted_z);
    evaluator.mod_switch_to_inplace(plain_125_mul_z_b,temp1.parms_id());
    evaluator.relinearize_inplace(temp1,relin_keys);
    evaluator.rescale_to_next_inplace(temp1);
    temp1.scale() =  scale;
    evaluator.add_inplace(result,temp1);

    return result;
}


// Dec and compute loss
void dec_compute_loss(SEALContext context, SecretKey secrect_key, Ciphertext encrypted_loss, size_t size){
    Plaintext plain_loss = Decrypt(context,secrect_key,encrypted_loss);
    vector<double> loss_vec;
    Decode(context,plain_loss,loss_vec);
    loss_vec.resize(size);
    double loss=0;
    for(size_t i=0; i<loss_vec.size(); ++i) loss+=loss_vec[i];
    loss = loss/double(size) + log(2);
    cout<<"************************************************loss: "<<loss<<"************************************************"<<endl;
}

// dec masked_encrypted_dJ_ab
vector<vector<double>> dec_dJ_ab(SEALContext context, SecretKey secrect_key, vector<Ciphertext> masked_encrypted_dJ_ab, size_t size){
    vector<vector<double>> result(masked_encrypted_dJ_ab.size(),vector<double>(size));
#pragma omp parallel for
    for(size_t i=0; i<masked_encrypted_dJ_ab.size(); ++i){
        vector<double> dJ_ab_i_vec;
        Plaintext plain_dJ_ab_i=Decrypt(context,secrect_key,masked_encrypted_dJ_ab[i]);
        Decode(context,plain_dJ_ab_i,dJ_ab_i_vec);
        dJ_ab_i_vec.resize(size);
        result[i] = dJ_ab_i_vec;
    }
    return result;
}

// revert dJ_ab and update weightsAB
void revert_dJ_ab_update_weightsAB(vector<double>& weightsAB, vector<vector<double>> mask_ab, vector<vector<double>> dJ_ab, double lr, size_t size, string client_name){
    cout<<client_name<<" weights:[";
    for(size_t i=0; i<dJ_ab.size(); ++i){
        double tempdJb=0;
        for(size_t j=0; j<dJ_ab[0].size(); ++j){
            tempdJb+=(dJ_ab[i][j]-mask_ab[i][j]);
        }
        weightsAB[i] -= lr*tempdJb/size;
        cout<<"\t"<<weightsAB[i];
    }
    cout<<"\t]\n";
}