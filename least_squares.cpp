#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
using namespace std;


// VERY FIRST THING TO DO IS DO DEFINE A MATRIX CLASS WITH NEEDED PROPERTIES


// A "matrix" object is an 2D array with known dimensions and functions to load and extract values from the cells.
class matrix{ 
    
    private:
    vector <vector <double> > data;

    public:
        int nrow;
        int ncol;

        // Initiate matrix as filled with zeros. If no dimensions provided, shape of matrix is 1x1.
        matrix(int number_of_rows = 0, int number_of_columns = 0){
            nrow = number_of_rows;
            ncol = number_of_columns;

            vector <vector <double> > dat;
            vector <double> row(number_of_columns, 0);

            for(int r = 0; r < number_of_rows; r++){
                dat.push_back(row);
            }
            data = dat;
        }

        // Returns value in the "row"-th row and "col"-th column of the matrix.
        double get(int row, int col){
            if(row >= nrow || col >= ncol || row < 0 || col < 0){
                throw out_of_range("Check matrix' dimensions.");
            }
            return data[row][col];
            }

        // Sets value in the "row"-th row and "col"-th column of the matrix as "value". 
        void set(int row, int col, double value){
             if(row >= nrow || col >= ncol || row < 0 || col < 0){
                throw out_of_range("Check matrix' dimensions.");
             }
            data[row][col] = value;
        }

        // Print the matrix.
        void print(){
            for(int r = 0; r < nrow; r++){
            for(int c = 0; c < ncol; c++){
                cout << data[r][c] << " ";
            }
            cout << endl;
            }
        }

        // To "row" add "added_row" multiplied by "scalar".
        void add_row(int row, int added_row, double scalar){  
            if( row < 0 || added_row < 0 || row >= nrow || added_row >= nrow){
                throw out_of_range("There exists no row of such an index.");
            }     
            for(int c = 0; c < ncol; c++){
            data[row][c] += scalar * data[added_row][c];
            }
        }

        // Replace "row1" with "row2".
        void replace_row(int row1, int row2){
            if( row1 < 0 || row2 < 0 || row1 >= nrow || row2 >= nrow){
                throw out_of_range("There exists no row of such an index.");
            }
            vector <double> temp_vector;

            for(int i = 0; i < ncol; i++){
            temp_vector.push_back(data[row1][i]);
            }
            for(int i = 0; i < ncol; i++){
                data[row1][i] = data[row2][i];
                data[row2][i] = temp_vector[i];
            }
        }

        // Fill matrix with low quality random integers in range 0-9.
        void random(){
            for(int r = 0; r < nrow; r++){
            for(int c = 0; c < ncol; c++){
                data[r][c] = rand() % 10;
            }}
        }
};

// A "CSV" object is basically a vector of variables' names and a matrix of numerical data.
class CSV{
    public:
    vector <string> names;
    matrix data;

    CSV(vector <string> names_of_variables, matrix values){
        names = names_of_variables;
        data = values;
    }
};

// The function to load CSV class object from .csv file.
CSV load_CSV(ifstream &CSV_file_name){

// We fill vector of variables' names

vector <string> variable_names;
string line, word;
getline(CSV_file_name, line);
string::iterator it;

// Read single "word" letter by letter omitting '"' and ' ' signs. When ',' reached, the word is added to "line" and new word is started.

for(it = line.begin(); it != line.end(); it++){
    if(*it == ','){
        variable_names.push_back(word);
        word.clear();
        }
    else if(*it != ' ' && *it != '\"'){ 
        word.push_back(*it);
        }
}
variable_names.push_back(word);
line.clear();

// We fill numerical 'matrix' of data. We do not use matrix class to represent data, because we don't know the size of it.

vector <vector <double> > data;
string number;
vector <double> row;

// Again we read numbers cipher by cipher, omitting ' ' and beginning new number when reaching ",".

while(getline(CSV_file_name, line)){
    for(it = line.begin(); it != line.end(); it++){
        if(*it == ','){
            row.push_back(stod(number));
            number.clear();
        }
        else if(*it != ' '){
            number.push_back(*it);
        }
    }
    row.push_back(stod(number));
    data.push_back(row);    // so data[i][j] means i-th observation of j-th variable
    row.clear();
    number.clear();
}

// Now we transform vector type data to matrix type data.

matrix D(data.size(), data.at(0).size());
for(int r = 0; r < D.nrow; r++){
for(int c = 0; c < D.ncol; c++){

D.set(r, c, data[r][c]);

}}

// Return output in a nicer way.

CSV output(variable_names, D);
return output;
}

// The function to save CSV class object to .csv file.
void save_CSV(CSV predictions){
    ofstream to_file;
    to_file.open("LinearReg predictions.csv");

    // Save variables' names

    for(int i = 0; i < predictions.names.size() - 1; i++){
        to_file << predictions.names[i] << ", ";
    }
    to_file << predictions.names[predictions.names.size() - 1] << endl;

    // Save numerical data

    for(int r = 0; r < predictions.data.nrow; r++){
        for(int c = 0; c < predictions.data.ncol - 1; c++){
            to_file << predictions.data.get(r, c) << ", ";
        }
        to_file << predictions.data.get(r, predictions.data.ncol - 1);
        if(r != predictions.data.nrow){
            to_file << endl;
            }
    }
    to_file.close();
}


// NOW LET'S DEFINE SOME BASIC OPERATIONS OF LINEAR ALGEBRA


// Returns result of matrix multiplication of matrices A and B.
matrix multiply(matrix A, matrix B){
   
    if(A.ncol != B.nrow){
        throw invalid_argument("Dimensions of the matrices do not tally.");
    }

    matrix AB(A.nrow, B.ncol);

    int times = B.nrow; // number of elements of dot product
    double sum = 0;

    for(int row_num = 0; row_num < AB.nrow; row_num++){
    for(int col_num = 0; col_num < AB.ncol; col_num++){
        for(int i = 0; i < times; i++){
            sum += (A.get(row_num, i) * B.get(i, col_num)); // here we multiply row by column element-wise
        }
        AB.set(row_num, col_num, sum);
        sum = 0;
    }}

    return AB;
}

// Transpose the matrix.
matrix transpose(matrix M){

    matrix M_trans(M.ncol, M.nrow);
    vector <double> row;

    for(int row_num = 0; row_num < M.nrow; row_num++){
    for(int col_num = 0; col_num < M.ncol; col_num++){
        M_trans.set(col_num, row_num, M.get(row_num, col_num));
    }}

    return M_trans;
}

/* The function finds the inverse matrix of input matrix using the method of Gauss elimination. If the matrix is not quadratic or is singular,
the exceptions are thrown.
 */ 
matrix inverse(matrix A){

    if(A.nrow != A.ncol){
        throw invalid_argument("Input matrix is not a quadratic one. There exists no inverse matrix.");
    }

    // We create an extended matrix by "glueing" to the right of matrix A an identity matrix.

    matrix A_ext(A.nrow, 2 * A.ncol);
    for(int r = 0; r < A.nrow; r++){
    for(int c = 0; c < A.ncol; c++){
        A_ext.set(r, c, A.get(r, c));
    }}
    for(int r = 0; r < A.nrow; r++){
        A_ext.set(r, r + A.nrow, 1); // ones on "second" diagonal
    }

    // Time for Gauss elimination. In each turn we transform one row of left matrix to be part of identity matrix.

    for(int turn = 0; turn < A_ext.nrow; turn++){

        // We replace rows as long as we don't get a row starting with non-zero number.

        int t = turn + 1;

        while(A_ext.get(turn, turn) == 0){
            A_ext.replace_row(turn, t);
            t++;

            if(t == A_ext.nrow){
                throw invalid_argument("Matrix is singular. There exists no inverse matrix.");
            }
        }
        // normalize row to be added.

        double norm = A_ext.get(turn, turn);
        for(int c = 0; c < A_ext.ncol; c++){
            A_ext.set(turn, c, A_ext.get(turn, c) / norm);
        }
        // subtract normalized row from the others

        for(int r = 0; r < A_ext.nrow; r++){
            if(r == turn){continue;}
            A_ext.add_row(r, turn, - A_ext.get(r, turn));
        }
    }

    // We extract from extended matrix the "right-side" matrix, which is wanted inverse matrix.

    matrix A_inv(A.nrow, A.ncol);
    for(int r = 0; r < A.nrow; r++){
    for(int c = 0; c < A.ncol; c++){
        A_inv.set(r, c, A_ext.get(r, c + A.ncol));
    }}

    return A_inv;
}

// The function returns value of R^2 statistics of model's fit.
double r_squared(vector <double> data_exp, vector <double> data_pred){

    double length = (double) data_exp.size();
    
    double mean = 0;
    for(int i = 0; i < length; i++){
        mean += data_exp[i];
    }
    mean /= length;

    double TSS = 0; // total sum of squares
    for(int i = 0; i < length; i++){
        TSS += pow(data_exp[i] - mean, 2);
    }

    double RSS = 0; // residual sum of squares
    for(int i = 0; i < length; i++){
        RSS += pow(data_exp[i] - data_pred[i], 2);
    }

    return 1 - RSS / TSS;
}

int main(){

// csv file loading

string file_name;
cout << "Enter name of file with train data." << endl << "> ";
cin >> file_name;
ifstream file;
file.open(file_name);
if(! file.good()){
    cout << "\nCould not open the file." << endl;
    system ("pause");
    return 0;
    }
else{
    cout << endl << "File opened successfully." << endl << endl;}

CSV loaded = load_CSV(file);
file.close();

// choosing variable to regress

cout << "Data file is set of observations of " << loaded.names.size() << " variables. Type \"yes\" if you wish to print variables' names or \"no\" otherwise." << endl;
string answer;
cout << ">";
cin >> answer;
while(answer != "yes" && answer != "no"){
    cout << "Command unknown." << endl << ">";
    cin >> answer;
}
if(answer == "yes"){
    cout << endl << "Variables' names:" << endl;
    for(int i = 0; i < loaded.names.size(); i++){
        cout << i + 1 << ". " << loaded.names[i] << endl;}
}

cout << endl << "Enter name of variable you wish to apply linear regression to. The regression will be performed with respect to all other variables." << endl << ">";
string wanted;
int index;
bool move_on = false;
while(! move_on){
    cin >> wanted;
    for(int i = 0; i < loaded.names.size(); i++){
        if(wanted == loaded.names.at(i)){
            index = i;
            move_on = true;
            break;
        }
    }
    if(! move_on){
        cout << "Variable not found. Try again." << endl << ">";
    }   
}

// create vector of names of all variables with respect to which the regression is performed

vector <string> other_names;

for(int i = 0; i < loaded.names.size(); i++){
    if(loaded.names.at(i) != wanted){
        other_names.push_back(loaded.names.at(i));
        }
}

// preparations of data to linear regression

matrix X(loaded.data.nrow, loaded.data.ncol);
matrix Y(loaded.data.nrow, 1);

// first we fill first column of "X" with ones

for(int r = 0; r < loaded.data.nrow; r++){
    X.set(r, 0, 1);
}

// now we copy "data" matrix omitting column of variable to be regressed

for(int colD = 0, colX = 1; colD < loaded.data.ncol; colD++, colX++){
    if(colD == index){
        colD += 1;
        if(colD == loaded.data.ncol){break;}
    }
    for(int r = 0; r < loaded.data.nrow; r++){
        X.set(r, colX, loaded.data.get(r, colD));
    }
}

// now we fill vector of variable to be regressed

for(int r = 0; r < loaded.data.nrow; r++){
    Y.set(r, 0, loaded.data.get(r, index));
}

// finally time for some math

matrix A = inverse(multiply(transpose(X),X));
matrix coef = multiply(multiply(A, transpose(X)), Y); // coef is a vertical vector 

// Now we calculate model predictions.

vector <double> Y_pred;
for(int r = 0; r < X.nrow; r++){
    double sum = coef.get(0, 0);
    for(int c = 1; c < X.ncol; c++){
        sum += coef.get(c, 0) * X.get(r, c);
    }
    Y_pred.push_back(sum);
}

// Time for testing if model describes data correctly.

vector <double> y(Y.nrow);
for(int i = 0; i < Y.nrow; i++){
    y[i] = Y.get(i, 0);
}

// Print data or exit.

cout << endl << "Regression ended with success. Value of R^2 statistics is " << r_squared(y, Y_pred) << ".";
cout << endl << "Do you wish to print found regression coefficients? (\"yes\", \"no\")." << endl;

string answer2;
cout << ">";
cin >> answer2;
while(answer2 != "yes" && answer2 != "no"){
    cout << "Command unknown." << endl << ">";
    cin >> answer2;
}
if(answer2 == "yes"){
    cout << endl << "[" << wanted << "] = ";
    for(int i = 0; i < other_names.size(); i++){
        cout << coef.get(i + 1, 0) << "*[" << other_names.at(i) << "] + ";
    }
    cout << coef.get(0, 0) << endl << endl;
}

cout << "Type \"save\", if you wish to save predicted values to .csv file. Otherwise type \"exit\"." << endl;

string answer3;
cout << ">";
cin >> answer3;
while(answer3 != "save" && answer3 != "exit"){
    cout << "Command unknown." << endl << ">";
    cin >> answer3;
}
if(answer3 == "save"){

    // Reformatting predictions.

    vector <string>  pred_names = loaded.names;
    pred_names.push_back(wanted + "_pred");

    matrix pred_data(loaded.data.nrow, loaded.data.ncol + 1);

    for(int r = 0; r < loaded.data.nrow; r++){
    for(int c = 0; c < loaded.data.ncol; c++){
        pred_data.set(r, c, loaded.data.get(r, c));
    }
        pred_data.set(r, loaded.data.ncol, Y_pred[r]);
    }

    // Save predictions to .csv.

    CSV ready_to_save(pred_names, pred_data);
    save_CSV(ready_to_save);

    cout << "Saving completed." << endl << endl;
    system("pause");
}

return 0;
}
