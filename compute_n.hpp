#include <vector>
using namespace std;

vector<double> compute_z_ab( vector<vector<double>> X, vector<double> weights, int size, double bias){
    vector<double> z_ab(size);
#pragma omp parallel for
    for(int i=0; i<size; ++i){
        double sum_temp=0;
        for(size_t j=0; j<weights.size(); ++j){
            sum_temp += X[i][j]*weights[j];
        }
        z_ab[i] = sum_temp + bias;
    }
    return z_ab;
}

vector<double> compute_u_a(vector<double> z_a, size_t size){
    vector<double> u_a(size);
#pragma omp parallel for
    for(size_t i=0; i<size; ++i){
        u_a[i]=0.25*z_a[i];
    }
    return u_a;
}
vector<double> compute_z_a_squre(vector<double> z_a, size_t size){
    vector<double> z_a_squre(size);
#pragma omp parallel for
    for(size_t i=0; i<size; ++i){
        z_a_squre[i]=z_a[i]*z_a[i];
    }
    return z_a_squre;
}

vector<double> compute_u_b(vector<double> y, vector<double> z_b, size_t size){
    vector<double> u_b(size);
#pragma omp parallel for
    for(size_t i=0; i<size; ++i){
        u_b[i] = 0.25*z_b[i]-y[i]+0.5;
    }
    return u_b;
}

vector<double> compute_u(vector<double> u_a, vector<double> u_b, size_t size){
    vector<double> u(size);
#pragma omp parallel for
    for(size_t i=0; i<size; ++i){
        u[i] = u_a[i]+u_b[i];
    }
    return u;
}

vector<double> compute_dJ( vector<vector<double>> X, vector<double> u, vector<double> weights, double lambda, size_t size){
    vector<double> dJ(weights.size());
#pragma omp parallel for
    for(size_t i=0; i<weights.size(); ++i){
        double temp_sum=0;
        for(size_t j=0; j<size; ++j){
            temp_sum += X[j][i]*u[j];
        }
        dJ[i] = temp_sum + lambda*weights[i];
    }
    return dJ;
}

vector<double> compute_z(vector<double> u_a, vector<double> z_b){
    vector<double> z(u_a.size());
#pragma omp parallel for
    for(size_t i=0; i<u_a.size(); ++i){
        z[i] = 4*u_a[i]+z_b[i];
    }
    return z;
}

double compute_loss(vector<double> z, vector<double> y, vector<double> z_a_squre, vector<double> z_b, vector<double> u_a){
    double loss = 0;
    for(size_t i=0; i<z.size(); ++i){
        loss += (0.5-y[i])*z[i] + 0.125*z_a_squre[i] +0.125*z_b[i]*(z[i]+4*u_a[i]);
    }
    return loss;
}

void update_weights( vector<double>& weights, vector<double> dJ, double lr, size_t size, string client_name){
    cout<<client_name<<" weights:[";
    for(size_t i=0; i<weights.size(); ++i){
        weights[i] -= lr*dJ[i]/size;
        cout<<"\t"<<weights[i];
    }
    cout<<"\t]"<<endl;
}

void compute_acc(vector<double> weightsA, vector<double> weightsB, vector<vector<double>> XA_test, vector<vector<double>> XB_test, vector<double> y_test){
    double countA = 0, countB = 0;
    for(size_t i=0; i<y_test.size(); ++i){
        double tempA = 0, tempB = 0;
        for(size_t j=0; j<weightsA.size(); ++j){
            tempA+=weightsA[j]*XA_test[i][j];
        }
        for(size_t k=0; k<weightsB.size(); ++k){
            tempB+=weightsB[k]*XB_test[i][k];
        }
        if((tempA>0&&y_test[i]==1)or(tempA<=0&&y_test[i]==0)) countA++;
        if((tempB>0&&y_test[i]==1)or(tempB<=0&&y_test[i]==0)) countB++;
        // countA+=((tempA>0)==(y_test[i]));
        // countB+=((tempB>0)==(y_test[i]));
    }

    cout<<"*********************** part A acc: "<<countA/(double)y_test.size()
    <<" ************************ part B acc: "<<countB/(double)y_test.size()
    <<" ************************"<<endl;
}