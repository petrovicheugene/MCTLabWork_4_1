//=====================================================
#include <iostream>
#include <vector>
#include <omp.h>
#include <fstream>
#include <string>
#include <regex>
#include <iomanip>
//=====================================================
using namespace std;

typedef vector<vector<double>> Matrix;

bool loadDataFromFile();
void trimLine(string& line);
void outputMatrices();
vector<double> parseLine(const string& line);
void calculateMatrixE();
void outputMatrixToFile(const Matrix& matrix);
void outputMatrixToDisplay(const Matrix& matrix);
void transponseSquareMatrix(Matrix& matrix);
void multiplyMatrixByFactor(Matrix& matrix, double factor);
bool inverseMatrix(Matrix matrix, Matrix& inverted_matrix);
Matrix matrixMinor(const Matrix& matrix, unsigned int r, unsigned int c);
double calcAlgComplement(const Matrix& matrix, unsigned int r, unsigned int c);
double determinant(Matrix matrix);
Matrix triangulateMatrix(const Matrix& matrix, int* p_swap_count);
unsigned int findMaxElementRow(Matrix matrix, unsigned int column);
//=====================================================
void addMatrices(Matrix& matrix_1,
    const Matrix& matrix_2);
Matrix multiplyMatrices(const Matrix& matrix_1,
    const Matrix& matrix_2);

static string inFileName = "./matrices_data.txt";
static string outFileName = "./matrix_E.txt";

static Matrix A;
static Matrix B;
static Matrix C;
static Matrix D;
static Matrix E;
//=====================================================
int main()
{
    cout << "Matrix operations." << endl;

    if (!loadDataFromFile())
    {
        system("pause");
        return 0;
    }

    outputMatrices();
    calculateMatrixE();

    cout << "*****" << endl;
    cout << "Result: Matrix E" << endl;
    outputMatrixToDisplay(E);
    outputMatrixToFile(E);

    system("pause");
    return 0;
}
//=====================================================
bool loadDataFromFile()
{
    cout << "Matrices initialization from file \"" << inFileName << "\"" << endl;
    cout << endl;
    // open file
    ifstream file(inFileName);
    if (!file.is_open())
    {
        cout << "Unable to open data file" << endl;
        return false;
    }

    string line;
    string matrixName;
    vector<double> values;
    while (getline(file, line))
    {
        trimLine(line);

        // Matrix identification
        if (line == "A")
        {
            matrixName = line;
            continue;
        }
        else if (line == "B")
        {
            matrixName = line;
            continue;
        }
        else if (line == "C")
        {
            matrixName = line;
            continue;
        }
        else if (line == "D")
        {
            matrixName = line;
            continue;
        }
        else if (line.empty())
        {
            matrixName.clear();
            continue;
        }

        // Matrix initialization
        values = parseLine(line);
        if (values.empty())
        {
            continue;
        }

        if (matrixName == "A")
        {
            A.push_back(values);
        }
        else if (matrixName == "B")
        {
            B.push_back(values);
        }
        else if (matrixName == "C")
        {
            C.push_back(values);
        }
        else if (matrixName == "D")
        {
            D.push_back(values);
        }
        // else if (matrixName.empty()) cntinue;
    }

    file.close();
    return true;
}
//=====================================================
void outputMatrixToFile(const Matrix& matrix)
{
    if (matrix.empty())
    {
        return;
    }

    ofstream file(outFileName);
    if (!file.is_open())
    {
        cout << "Unable to open data file" << endl;
        return;
    }

    file << "Matrix E" << endl;
    file << fixed << setprecision(2);
    for (unsigned int r = 0; r < E.size(); r++)
    {
        for (unsigned int c = 0; c < E.size(); c++)
        {
            file << E[r][c] << "\t";
        }
        file << endl;
    }

    file.close();
}
//=====================================================
void trimLine(string& line)
{
    line.erase(line.find_last_not_of(" \n\r\t") + 1);
    line.erase(0, line.find_first_not_of(" "));
}
//=====================================================
vector<double> parseLine(const string& line)
{
    std::regex regex{ R"([\s,]+)" }; // split by space and comma
    std::sregex_token_iterator it{ line.begin(), line.end(), regex, -1 };
    std::vector<std::string> words{ it, {} };
    vector<double> values;

    for (size_t i = 0; i < words.size(); i++)
    {
        values.push_back(stod(words.at(i)));
    }
    return values;
}
//=====================================================
void outputMatrices()
{
    cout << "Matrix A" << endl;
    outputMatrixToDisplay(A);

    cout << "Matrix B" << endl;
    outputMatrixToDisplay(B);

    cout << "Matrix C" << endl;
    outputMatrixToDisplay(C);

    cout << "Matrix D" << endl;
    outputMatrixToDisplay(D);
}
//=====================================================
void calculateMatrixE()
{
    Matrix invertedC;
    if (!inverseMatrix(C, invertedC))
    {
        cout << "Error: cannot inverce matrix C." << endl;
        return;
    }

    transponseSquareMatrix(D);
    multiplyMatrixByFactor(D, -1);
    addMatrices(invertedC, D); // result in invertedC

    E = multiplyMatrices(A, B);
    addMatrices(E, C);
}
//=====================================================
void outputMatrixToDisplay(const Matrix& matrix)
{
    if (matrix.empty())
    {
        return;
    }

    unsigned int rows = matrix.size();
    unsigned int columns = matrix.at(0).size();

    for (unsigned int row = 0; row < rows; row++)
    {
        for (unsigned int col = 0; col < columns; col++)
        {
            printf("%9.2f", matrix[row][col]);
        }
        cout << endl;
    }
    cout << endl;
}
//=====================================================
void transponseSquareMatrix(Matrix& matrix)
{
#pragma omp parallel
    {
#pragma omp for
        for (unsigned int r = 0; r < matrix.size(); r++)
        {
            for (unsigned int c = r + 1; c < matrix.size(); c++)
            {
                swap(matrix[r][c], matrix[c][r]);
            }
        }
    }
}
//=====================================================
bool inverseMatrix(Matrix matrix, Matrix& inverted_matrix)
{
    double det = determinant(matrix);
    if (det == 0.0)
    {
        cout << "Error of matrix inversion. Determinant is null." << endl;
        return false;
    }

    transponseSquareMatrix(matrix);

    // create new matrix the same to transponsed one
    inverted_matrix = matrix;
    unsigned int matrix_size = matrix.size();
    // replace every element by it's own AlgComplement
#pragma omp parallel
    {
#pragma omp for
        for (unsigned int r = 0; r < matrix_size; r++)
        {
            for (unsigned int c = 0; c < matrix_size; c++)
            {
                inverted_matrix[r][c] = calcAlgComplement(matrix, r, c);
            }
        }
    }
    // multiply newMatrix by 1/det
    double factor = 1 / det;
    multiplyMatrixByFactor(inverted_matrix, factor);

    return true;
}
//=====================================================
Matrix matrixMinor(const Matrix& matrix, unsigned int mr, unsigned int mc)
{
    unsigned int matrix_size = matrix.size();
    Matrix minor;
    for (unsigned int r = 0; r < matrix_size; r++)
    {
        if (r == mr) continue;
        minor.push_back(vector<double>());
        for (unsigned int c = 0; c < matrix_size; c++)
        {
            if (c == mc) continue;
            minor[minor.size() - 1].push_back(matrix[r][c]);
        }
    }
    return minor;
}
//=====================================================
double calcAlgComplement(const Matrix& matrix, unsigned int r, unsigned int c)
{
    Matrix minor = matrixMinor(matrix, r, c);
    double det = determinant(minor);
    return det * pow(-1, static_cast<double>(r + c));
}
//=====================================================
double determinant(Matrix matrix) // matrix should be square
{
    int swapCount = 0;
    Matrix newMatrix = triangulateMatrix(matrix, &swapCount);
    unsigned int matrix_size = newMatrix.size();

    // calc determinant
    double det = swapCount % 2 ? 1.0 : -1.0;
    bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
    {
#pragma omp for reduction (*: det)
        for (unsigned int i = 0; i < matrix_size; i++)
        {
            det *= newMatrix[i][i];
        }
    }
    return det;
}
//=====================================================
Matrix triangulateMatrix(const Matrix& matrix, int* pSwapCount)
{
    // create and init new Matrix
    Matrix new_matrix = matrix;

    unsigned int matrix_size = new_matrix.size();
    unsigned int matrix_column_size = new_matrix[0].size();

    int swapCount = 0;
    for (unsigned int i = 0; i < matrix_size - 1; i++)
    {
        unsigned int maxElementRow = findMaxElementRow(new_matrix, i);

        if (i != maxElementRow)
        {
            swap(new_matrix[i], new_matrix[maxElementRow]);
            ++swapCount;
        }

        bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
        {
#pragma omp for
            for (unsigned int row = i + 1; row < matrix_size; row++)
            {
                double f = -new_matrix[row][i] / new_matrix[i][i];
                for (unsigned int k = i; k < matrix_column_size; ++k)
                {
                    new_matrix[row][k] += new_matrix[i][k] * f;
                }
            }
        }
    }
    if (pSwapCount != nullptr)
    {
        *pSwapCount = swapCount;
    }
    return new_matrix;
}
//=====================================================
unsigned int findMaxElementRow(Matrix matrix, unsigned int column)
{
    unsigned int max_row_index = column;
    double max = abs(matrix[column][column]);
    unsigned int matrix_size = matrix.size();
    bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
    {
        double local_max = max;
        unsigned int local_max_row_index = max_row_index;
#pragma omp for
        for (unsigned int i = column + 1; i < matrix_size; i++)
        {
            double value = abs(matrix[i][column]);
            if (value > max)
            {
                local_max = value;
                local_max_row_index = i;
            }
        }

#pragma omp critical
        {
            if (max < local_max)
            {
                max = local_max;
                max_row_index = local_max_row_index;
            }
        }
    }
    return max_row_index;
}
//=====================================================
Matrix multiplyMatrices(const Matrix& matrix_1,
    const Matrix& matrix_2) // required condition: matrix_1 col count == matrix_2 row count
{
    // init new matrix
    Matrix new_matrix;
    bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
    {
        double c = 0.0;
#pragma omp for ordered
        for (unsigned int m_1r = 0; m_1r < matrix_1.size(); m_1r++)
        {
            vector<double> row;
            for (unsigned int m_2c = 0; m_2c < matrix_2.at(0).size(); m_2c++)
            {
                c = 0;
                for (unsigned int k = 0; k < matrix_2.size(); k++)
                {
                    c += matrix_1[m_1r][k] * matrix_2[k][m_2c];
                }

                row.push_back(c);
            }
#pragma omp ordered
            new_matrix.push_back(row);

        }
    }
    return new_matrix;
}
//=====================================================
void multiplyMatrixByFactor(Matrix& matrix, double factor)
{
    bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
    {
#pragma omp for
        for (unsigned int r = 0; r < matrix.size(); r++)
        {
            for (unsigned int c = 0; c < matrix.at(r).size(); c++)
            {
                matrix[r][c] *= factor;
            }

        }
    }
}
//=====================================================
void addMatrices(Matrix& matrix_1, const Matrix& matrix_2) // matrices should have the same size
{
    bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
    {
        #pragma omp for
        for (unsigned int r = 0; r < matrix_1.size(); r++)
        {
            for (unsigned int c = 0; c < matrix_1.at(r).size(); c++)
            {
                matrix_1[r][c] += matrix_2[r][c];
            }
        }
    }
}
//=====================================================
