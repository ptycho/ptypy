#include <gtest/gtest.h>

#include "utils/Indexing.h"

TEST(IndexingTest, inRange)
{
  IndexReflect ind(1, 10);
  EXPECT_EQ(1, ind(1));
  EXPECT_EQ(5, ind(5));
  EXPECT_EQ(3, ind(3));
  EXPECT_EQ(9, ind(9));
}

TEST(IndexingTest, reflectTop1)
{
  IndexReflect ind(1, 10);
  EXPECT_EQ(9, ind(10));
  EXPECT_EQ(8, ind(11));
  EXPECT_EQ(7, ind(12));
  EXPECT_EQ(6, ind(13));
}

TEST(IndexingTest, reflectTop0)
{
  IndexReflect ind(0, 10);
  EXPECT_EQ(9, ind(10));
  EXPECT_EQ(8, ind(11));
  EXPECT_EQ(7, ind(12));
  EXPECT_EQ(6, ind(13));
}

TEST(IndexingTest, reflectTopNeg)
{
  IndexReflect ind(-5, 0);
  EXPECT_EQ(-2, ind(1));
  EXPECT_EQ(-3, ind(2));
  EXPECT_EQ(-4, ind(3));
  EXPECT_EQ(-5, ind(4));
}

TEST(IndexingTest, reflectBottom)
{
  IndexReflect ind(1, 10);
  EXPECT_EQ(1, ind(0));
  EXPECT_EQ(2, ind(-1));
  EXPECT_EQ(3, ind(-2));
}

TEST(IndexingTest, reflectBottom0)
{
  IndexReflect ind(0, 10);
  EXPECT_EQ(0, ind(0));
  EXPECT_EQ(0, ind(-1));
  EXPECT_EQ(1, ind(-2));
  EXPECT_EQ(2, ind(-3));
}

TEST(IndexingTest, reflectBottomNeg)
{
  IndexReflect ind(-5, -1);
  EXPECT_EQ(-5, ind(-5));
  EXPECT_EQ(-5, ind(-6));
  EXPECT_EQ(-4, ind(-7));
  EXPECT_EQ(-3, ind(-8));
}

TEST(IndexingTest, reflectTopWide)
{
  IndexReflect ind(1, 3);
  EXPECT_EQ(2, ind(3));
  EXPECT_EQ(1, ind(4));
  EXPECT_EQ(1, ind(5));
  EXPECT_EQ(2, ind(6));
}

TEST(IndexingTest, reflectTopWideNeg)
{
  IndexReflect ind(-4, -1);
  EXPECT_EQ(-3, ind(0));
  EXPECT_EQ(-4, ind(1));
  EXPECT_EQ(-4, ind(2));
  EXPECT_EQ(-3, ind(3));
  EXPECT_EQ(-2, ind(4));
  EXPECT_EQ(-2, ind(5));
}

TEST(IndexingTest, reflectBottomWide)
{
  IndexReflect ind(1, 3);
  EXPECT_EQ(2, ind(-2));
  EXPECT_EQ(1, ind(-3));
  EXPECT_EQ(1, ind(-4));
  EXPECT_EQ(2, ind(-5));
}