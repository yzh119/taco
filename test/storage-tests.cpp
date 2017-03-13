#include "test.h"
#include "test_tensors.h"

#include <map>

#include "tensor.h"
#include "format.h"
#include "storage/storage.h"
#include "util/strings.h"

typedef int                     IndexType;
typedef std::vector<IndexType>  IndexArray; // Index values
typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
typedef std::vector<Index>      Indices;    // One Index per level

using taco::Tensor;
using taco::Format;
using taco::LevelType;

const auto Dense = taco::LevelType::Dense;
const auto Sparse = taco::LevelType::Sparse;
const auto Fixed = taco::LevelType::Fixed;

struct TestData {
  TestData(Tensor<double> tensor,
           const Indices& expectedIndices,
           const vector<double> expectedValues)
      : tensor(tensor),
        expectedIndices(expectedIndices), expectedValues(expectedValues) {
  }

  Tensor<double> tensor;
  Indices        expectedIndices;
  vector<double> expectedValues;
};

static ostream &operator<<(ostream& os, const TestData& data) {
  os << taco::util::join(data.tensor.getDimensions(), "x")
     << " (" << data.tensor.getFormat() << ")";
  return os;
}

struct storage : public TestWithParam<TestData> {};

TEST_P(storage, pack) {
  Tensor<double> tensor = GetParam().tensor;

  auto storage = tensor.getStorage();
  ASSERT_TRUE(storage.defined());
  auto levels = storage.getFormat().getLevels();

  // Check that the indices are as expected
  auto& expectedIndices = GetParam().expectedIndices;
  auto size = storage.getSize();

  for (size_t i=0; i < levels.size(); ++i) {
    auto expectedIndex = expectedIndices[i];
    auto levelIndex = storage.getLevelIndex(i);
    auto levelIndexSize = size.levelIndices[i];

    switch (levels[i].getType()) {
      case LevelType::Dense: {
        iassert(expectedIndex.size() == 1);
        ASSERT_ARRAY_EQ(expectedIndex[0], {levelIndex.ptr, levelIndexSize.ptr});
        ASSERT_EQ(nullptr, levelIndex.idx);
        ASSERT_EQ(0u, levelIndexSize.idx);
        break;
      }
      case LevelType::Sparse:
      case LevelType::Fixed: {
        iassert(expectedIndex.size() == 2);
        ASSERT_ARRAY_EQ(expectedIndex[0], {levelIndex.ptr, levelIndexSize.ptr});
        ASSERT_ARRAY_EQ(expectedIndex[1], {levelIndex.idx, levelIndexSize.idx});
        break;
      }
      case LevelType::Offset:
      case LevelType::Replicated:
        break;
    }
  }

  auto& expectedValues = GetParam().expectedValues;
  ASSERT_EQ(expectedValues.size(), storage.getSize().values);
  ASSERT_ARRAY_EQ(expectedValues, {storage.getValues(), size.values});
}

INSTANTIATE_TEST_CASE_P(scalar, storage,
    Values(TestData(da("a", Format()),
                    {
                    },
                    {2}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(vector, storage,
    Values(TestData(d1a("a", Format({Dense})),
                    {
                      {
                        // Dense index
                        {1}
                      }
                    },
                    {2}
                    ),
           TestData(d1a("a", Format({Sparse})),
                    {
                      {
                        // Sparse index
                        {0,1},
                        {0}
                      }
                    },
                    {2}
                    ),
            TestData(d1a("a", Format({Fixed})),
                    {
                      {
                        // Fixed index
                        {1},
                        {0}
                      }
                    },
                    {2}
                    ),
           TestData(d5a("a", Format({Dense})),
                    {
                      {
                        // Dense index
                        {5}
                      }
                    },
                    {0, 2, 0, 0, 3}
                    ),
           TestData(d5a("a", Format({Sparse})),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {1,4}
                      },
                    },
                    {2, 3}
                    ),
            TestData(d5a("a", Format({Fixed})),
                    {
                      {
                        // Sparse index
                        {2},
                        {1,4}
                      },
                    },
                    {2, 3}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix, storage,
    Values(TestData(d33a("A", Format({Dense,Dense})),
                    {
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {0, 2, 0,
                     0, 0, 0,
                     3, 0, 4}
                    ),
           TestData(d33a("A", Format({Sparse,Dense})),  // Blocked svec
                    {
                      {
                        // Sparse index
                        {0, 2},
                        {0, 2},
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {0, 2, 0,
                     3, 0, 4}
                    ),
           TestData(d33a("A", Format({Dense,Sparse})),  // CSR
                    {
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Sparse index
                        {0, 1, 1, 3},
                        {1, 0, 2},
                      }
                    },
                    {2, 3, 4}
                    ),
           TestData(d33a("A", Format({Sparse,Sparse})),  // DCSR
                    {
                      {
                        // Sparse index
                        {0, 2},
                        {0, 2},
                      },
                      {
                        // Sparse index
                        {0, 1, 3},
                        {1, 0, 2},
                      }
                    },
                    {2, 3, 4}
                    )
)
);

INSTANTIATE_TEST_CASE_P(fixed, storage,
    Values(
        TestData(d33a("A", Format({Dense,Dense})),
                 {
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {0, 2, 0,
                  0, 0, 0,
                  3, 0, 4}
        ),
        TestData(d33a("A", Format({Dense,Fixed})),  // ELL
                 {
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Fixed index
                         {2},
                         {1, 1, 0, 0, 0, 2},
                     }
                 },
                 {2, 0, 0, 0, 3, 4}
        ),
        TestData(d33a("A", Format({Fixed,Dense})),
                 {
                     {
                         // Fixed index
                         {2},
                         {0, 2},
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {0, 2, 0, 3, 0, 4}
        ),
        TestData(d33at("A", Format({Dense,Dense})),
                 {
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {0, 0, 3,
                  2, 0, 0,
                  0, 0, 4}
        ),
        TestData(d33a("A", Format({Fixed,Sparse})),
                 {
                     {
                         // Fixed index
                         {2},
                         {0, 2},
                     },
                     {
                         // Sparse index
                         {0, 1, 3},
                         {1, 0, 2},
                     }
                 },
                 {2, 3, 4}
        ),
        TestData(d33a("A", Format({Sparse,Fixed})),
                 {
                     {
                         // Sparse index
                         {0, 2},
                         {0, 2},
                     },
                     {
                         // Fixed index
                         {2},
                         {1, 1, 0, 2}
                     }
                 },
                 {2, 0, 3, 4}
        ),
        TestData(d33a("A", Format({Dense,Fixed},{1,0})),
                 {
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Fixed index
                         {1},
                         {2, 0, 2},
                     }
                 },
                 {3, 2, 4}
        ),
        TestData(d33a("A", Format({Fixed,Dense},{1,0})),
                 {
                     {
                         // Fixed index
                         {3},
                         {0, 1, 2},
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {0, 0, 3, 2, 0, 0, 0, 0, 4}
        ),
        TestData(d33a("A", Format({Fixed,Fixed})),
                 {
                     {
                         // Fixed index
                         {2},
                         {0, 2},
                     },
                     {
                         // Fixed index
                         {2},
                         {1, 1, 0, 2},
                     }
                 },
                 {2, 0, 3, 4}
        ),
        TestData(d33a("A", Format({Fixed,Fixed}, {1,0})),
                 {
                     {
                         // Fixed index
                         {3},
                         {0, 1, 2},
                     },
                     {
                         // Fixed index
                         {1},
                         {2, 0, 2},
                     }
                 },
                 {3, 2, 4}
        ),
        TestData(d233a("A", Format({Dense,Dense,Dense})),
                 {
                     {
                         // Dense index
                         {2}
                     },
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {2, 3, 0,
                  0, 0, 0,
                  0, 0, 4,

                  0, 5, 0,
                  0, 0, 0,
                  6, 0, 7}
        ),
        TestData(d233a("A", Format({Fixed,Dense,Dense})),
                 {
                     {
                         // Fixed index
                         {2},
                         {0,1}
                     },
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {2, 3, 0,
                  0, 0, 0,
                  0, 0, 4,

                  0, 5, 0,
                  0, 0, 0,
                  6, 0, 7}
        ),
        TestData(d233a("A", Format({Dense,Dense,Fixed})),
                 {
                     {
                         // Dense index
                         {2}
                     },
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 1, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2}
                     }
                 },
                 {2, 3, 0, 0, 4, 0, 5, 0, 0, 0, 6, 7}
        ),
        TestData(d233a("A", Format({Dense,Fixed,Dense})),
                 {
                     {
                         // Dense index
                         {2}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 2, 0, 2}
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {2, 3, 0, 0, 0, 4, 0, 5, 0, 6, 0, 7}
        ),
        TestData(d233a("A", Format({Fixed,Dense,Fixed})),
                 {
                     {
                         // Fixed index
                         {2},
                         {0,1}
                     },
                     {
                         // Dense index
                         {3}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 1, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2}
                     }
                 },
                 {2, 3, 0, 0, 4, 0, 5, 0, 0, 0, 6, 7}
        ),
        TestData(d233a("A", Format({Fixed,Fixed,Dense})),
                 {
                     {
                         // Fixed index
                         {2},
                         {0,1}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 2, 0, 2}
                     },
                     {
                         // Dense index
                         {3}
                     }
                 },
                 {2, 3, 0, 0, 0, 4, 0, 5, 0, 6, 0, 7}
        ),
        TestData(d233a("A", Format({Dense,Fixed,Fixed})),
                 {
                     {
                         // Dense index
                         {2}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 2, 0, 2}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 1, 2, 2, 1, 1, 0, 2}
                     }
                 },
                 {2, 3, 4, 0, 5, 0, 6, 7}
        ),
        TestData(d233a("A", Format({Fixed,Fixed,Fixed})),
                 {
                     {
                         // Fixed index
                         {2},
                         {0, 1}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 2, 0, 2}
                     },
                     {
                         // Fixed index
                         {2},
                         {0, 1, 2, 2, 1, 1, 0, 2}
                     }
                 },
                 {2, 3, 4, 0, 5, 0, 6, 7}
        )
    )
);

INSTANTIATE_TEST_CASE_P(matrix_blocked, storage,
    Values(TestData(d3322a("A", Format({Dense,Sparse,Dense,Dense})),
                    {
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Sparse index
                        {0,1,1,3},
                        {1,0,2}
                      },
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Dense index
                        {2}
                      }
                    },
                    {2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(matrix_col, storage,
    Values(TestData(d33a("A", Format({Dense,Dense}, {1,0})),
                    {
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {0, 0, 3,
                     2, 0, 0,
                     0, 0, 4}
                    ),
           TestData(d33a("A", Format({Sparse,Dense}, {1,0})),  // Blocked svec
                    {
                      {
                        // Sparse index
                        {0, 3},
                        {0, 1, 2},
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {0, 0, 3,
                     2, 0, 0,
                     0, 0, 4}
                    ),
           TestData(d33a("A", Format({Dense,Sparse}, {1,0})),  // CSC
                    {
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Sparse index
                        {0, 1, 2, 3},
                        {2, 0, 2},
                      }
                    },
                    {3, 2, 4}
                    ),
           TestData(d33a("A", Format({Sparse,Sparse}, {1,0})),  // DCSC
                    {
                      {
                        // Sparse index
                        {0, 3},
                        {0, 1, 2},
                      },
                      {
                        // Sparse index
                        {0, 1, 2, 3},
                        {2, 0, 2},
                      }
                    },
                    {3, 2, 4}
                    )
           )
);

INSTANTIATE_TEST_CASE_P(tensor3, storage,
    Values(TestData(d233a("A", Format({Dense,Dense,Dense})),
                    {
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {2, 3, 0,
                     0, 0, 0,
                     0, 0, 4,

                     0, 5, 0,
                     0, 0, 0,
                     6, 0, 7}
                    ),
           TestData(d233a("A", Format({Sparse,Dense,Dense})),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {0,1}
                      },
                      {
                        // Dense index
                        {3}
                      } ,
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {2, 3, 0,
                     0, 0, 0,
                     0, 0, 4,

                     0, 5, 0,
                     0, 0, 0,
                     6, 0, 7}
                    ),
           TestData(d233a("A", Format({Dense,Sparse,Dense})),
                    {
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Sparse index
                        {0,2,4},
                        {0,2,0,2}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {2, 3, 0,
                     0, 0, 4,

                     0, 5, 0,
                     6, 0, 7}
                    ),
           TestData(d233a("A", Format({Sparse,Sparse,Dense})),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {0,1}
                      },
                      {
                        // Sparse index
                        {0,2,4},
                        {0,2,0,2}
                      },
                      {
                        // Dense index
                        {3}
                      }
                    },
                    {2, 3, 0,
                     0, 0, 4,

                     0, 5, 0,
                     6, 0, 7}
                    ),
           TestData(d233a("A", Format({Dense,Dense,Sparse})),
                    {
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Sparse index
                        {0,2,2,3,4,4,6},
                        {0,1,2, 1,0,2}
                      }
                    },
                    {2, 3, 4, 5, 6, 7}
                    ),
           TestData(d233a("A", Format({Sparse,Dense,Sparse})),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {0,1}
                      },
                      {
                        // Dense index
                        {3}
                      },
                      {
                        // Sparse index
                        {0,2,2,3,4,4,6},
                        {0,1,2, 1,0,2}
                      }
                    },
                    {2, 3, 4, 5, 6, 7}
                    ),
           TestData(d233a("A", Format({Dense,Sparse,Sparse})),
                    {
                      {
                        // Dense index
                        {2}
                      },
                      {
                        // Sparse index
                        {0,2,4},
                        {0,2,0,2}
                      },
                      {
                        // Sparse index
                        {0,2,3,4,6},
                        {0,1,2, 1,0,2}
                      }
                    },
                    {2, 3, 4, 5, 6, 7}
                    ),
           TestData(d233a("A", Format({Sparse,Sparse,Sparse})),
                    {
                      {
                        // Sparse index
                        {0,2},
                        {0,1}
                      },
                      {
                        // Sparse index
                        {0,2,4},
                        {0,2,0,2}
                      },
                      {
                        // Sparse index
                        {0,2,3,4,6},
                        {0,1,2, 1,0,2}
                      }
                    },
                    {2, 3, 4, 5, 6, 7}
                    )
           )
);
