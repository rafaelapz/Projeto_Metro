
import numpy as np
import pandas as pd

csv_file = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/zonas_VIAGENS_MOTORIZADAS_SOMENTE.csv'
npy_file = '/Users/ellazyngier/Documents/github/tccII/scripts/dados/travel_matrix_VIAGENS_MOTORIZADAS_SOMENTE.npy'

print("=" * 80)
print("CSV TO NPY CONVERTER WITH DEBUGGING")
print("=" * 80)

print("\nLoading CSV with pandas...")


df = pd.read_csv(csv_file, header=None, thousands=',')
print(f"Initial DataFrame shape: {df.shape}")
print(f"DataFrame dtypes sample: {df.dtypes[:5].tolist()}")


print("\nSample of raw data (first 3x3):")
print(df.iloc[:3, :3])


print("\nConverting to numeric...")
conversion_issues = []
for col in df.columns:
    before_dtype = df[col].dtype
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    after_dtype = df[col].dtype
    
    # Check if any values were coerced
    if df[col].isna().any():
        conversion_issues.append(col)

if conversion_issues:
    print(f"WARNING: {len(conversion_issues)} columns had conversion issues")
else:
    print("✓ All columns converted successfully")


matrix = df.values.astype(np.float64)

print("\n" + "=" * 80)
print("MATRIX STATISTICS")
print("=" * 80)

print(f"\nBasic info:")
print(f"  Shape: {matrix.shape}")
print(f"  Data type: {matrix.dtype}")
print(f"  Size (total elements): {matrix.size:,}")


total_method1 = np.sum(matrix)
total_method2 = matrix.sum()
total_method3 = sum(matrix.flatten())

print(f"\n🔍 TOTAL TRIPS (using different methods):")
print(f"  np.sum(matrix):        {total_method1:,.0f}")
print(f"  matrix.sum():          {total_method2:,.0f}")
print(f"  sum(flatten):          {total_method3:,.0f}")

if total_method1 == total_method2 == total_method3:
    print("  ✓ All methods agree")
else:
    print("  ⚠️ WARNING: Methods disagree!")

# More detailed statistics
print(f"\nValue distribution:")
print(f"  Min value: {np.min(matrix):,.0f}")
print(f"  Max value: {np.max(matrix):,.0f}")
print(f"  Mean value: {np.mean(matrix):,.2f}")
print(f"  Median value: {np.median(matrix):,.0f}")
print(f"  Standard deviation: {np.std(matrix):,.2f}")

# Check for special values
print(f"\nData integrity checks:")
print(f"  Non-zero cells: {np.count_nonzero(matrix):,} ({np.count_nonzero(matrix)/matrix.size*100:.1f}%)")
print(f"  Zero cells: {(matrix == 0).sum():,} ({(matrix == 0).sum()/matrix.size*100:.1f}%)")
print(f"  Negative values: {(matrix < 0).sum():,}")
print(f"  NaN values: {np.isnan(matrix).sum():,}")
print(f"  Inf values: {np.isinf(matrix).sum():,}")

# Check symmetry (important for O-D matrices)
is_symmetric = np.allclose(matrix, matrix.T)
print(f"  Symmetric matrix: {is_symmetric}")
if not is_symmetric:
    diff_count = np.sum(matrix != matrix.T)
    max_diff = np.max(np.abs(matrix - matrix.T))
    print(f"    - Cells that differ: {diff_count:,}")
    print(f"    - Max difference: {max_diff:,.0f}")

# Row and column sums
row_sums = np.sum(matrix, axis=1)
col_sums = np.sum(matrix, axis=0)

print(f"\nRow/Column analysis:")
print(f"  Total of all row sums: {np.sum(row_sums):,.0f}")
print(f"  Total of all column sums: {np.sum(col_sums):,.0f}")
print(f"  Row with max trips: Zone {np.argmax(row_sums) + 1} ({np.max(row_sums):,.0f} trips)")
print(f"  Column with max trips: Zone {np.argmax(col_sums) + 1} ({np.max(col_sums):,.0f} trips)")
print(f"  Rows with zero trips: {(row_sums == 0).sum()}")
print(f"  Columns with zero trips: {(col_sums == 0).sum()}")

# Check specific values
print(f"\nChecking specific positions:")
positions = [(0,0), (0,1), (1,1), (524,524), (525,525), (526,526)]
for i, j in positions:
    print(f"  [{i:3},{j:3}]: {matrix[i,j]:>12,.0f}")

# Sample of the matrix
print(f"\nSample of matrix (top-left 5x5):")
for i in range(min(5, matrix.shape[0])):
    row_str = "  "
    for j in range(min(5, matrix.shape[1])):
        row_str += f"{matrix[i,j]:>10,.0f} "
    print(row_str)

# Save as NPY
np.save(npy_file, matrix)
print(f"\n" + "=" * 80)
print(f"✓ Saved to: {npy_file}")
print(f"✓ Matrix is 527x527 where row/col 0 = Zone 1, row/col 526 = Zone 527")

# Final verification: Load and check
print(f"\nVerifying saved file...")
loaded_matrix = np.load(npy_file)
loaded_total = np.sum(loaded_matrix)

print(f"  Loaded matrix shape: {loaded_matrix.shape}")
print(f"  Loaded matrix total: {loaded_total:,.0f}")

if loaded_total == total_method1:
    print(f"  ✓ Loaded total matches original ({loaded_total:,.0f})")
else:
    print(f"  ⚠️ WARNING: Loaded total differs!")
    print(f"     Original: {total_method1:,.0f}")
    print(f"     Loaded:   {loaded_total:,.0f}")
    print(f"     Diff:     {loaded_total - total_method1:,.0f}")

print("=" * 80)
print(f"FINAL TOTAL: {total_method1:,.0f} trips")
print("=" * 80)