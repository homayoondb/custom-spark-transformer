# Custom Spark Transformers

A collection of custom PySpark ML transformers that demonstrate how to extend Spark's ML pipeline capabilities with custom transformation logic.

## Features

This library includes three example transformers of increasing complexity:

1. **CustomImputer**: A basic transformer that handles missing values
   - Replaces nulls in string columns with 'none'
   - Replaces nulls in numeric columns with -99
   - Demonstrates basic transformation without fitting

2. **CustomAdder**: An intermediate transformer for feature engineering
   - Combines multiple numeric columns through addition
   - Shows how to work with multiple input columns
   - Creates derived features

3. **TargetEncoder**: An advanced transformer implementing the full estimator-transformer pattern
   - Learns mean statistics during fit()
   - Applies learned encodings during transform()
   - Demonstrates stateful transformations
   - Includes separate Model class for serialization

## Usage

```python
from pyspark.ml import Pipeline
from custom_spark_transformers import CustomImputer, CustomAdder, TargetEncoder

# Create transformers
imputer = CustomImputer()
adder = CustomAdder(inputCols=["col1", "col2"], outputCol="sum")
encoder = TargetEncoder(inputCol="category", targetCol="y", outputCol="encoded")

# Build pipeline
pipeline = Pipeline(stages=[
    imputer,
    adder,
    encoder
])

# Fit and transform
model = pipeline.fit(train_df)
result_df = model.transform(test_df)
```

## Requirements

- PySpark 3.x
- Python 3.7+

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/custom_spark_transformers.git
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
