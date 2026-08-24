#include <gtest/gtest.h>
#include "matrix.cuh"

TEST(MatrixBasics, CopyRoundTrip)
{
    Matrix<Location::Host, Layout::RowMajor> A(2, 3);
    int v = 0;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            A[{r, c}] = static_cast<float>(++v);

    auto A_d = A.copyToDevice(false);
    auto A_back = A_d.copyToHost(false);

    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
        {
            float a = A[{r, c}];
            float b = A_back[{r, c}];
            EXPECT_FLOAT_EQ(a, b);
        }
}


class TransposeSizes : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(TransposeSizes, RoundTrip)
{
    auto [r,c] = GetParam();
    Matrix<Location::Host, Layout::RowMajor>A(r, c);
    int v = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            A[{i, j}] = ++v;
    auto AT = A.Transposed();
    auto ATT = AT.Transposed();
    EXPECT_TRUE(ATT==A);
}

INSTANTIATE_TEST_SUITE_P(Sizes, TransposeSizes, ::testing::Values(std::make_tuple(1,1),std::make_tuple(4,59)));