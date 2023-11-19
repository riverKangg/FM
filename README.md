# FactorizationRec

AwesomeRecLib is a library that provides various algorithms for recommendation systems. It currently includes
implementations of FM (Factorization Machines), DeepFM, and xDeepFM.

## Installation

To install the library, use the following command:

```bash
pip install factorization-rec
```

## Usage Examples

### 1. FM (Factorization Machines)

```python
from factorization_rec import FM

# Load and preprocess data
# ...

# Create FM model
fm_model = FM()

# Train the model
fm_model.fit(train_data)

# Make predictions
predictions = fm_model.predict(test_data)
```

### 2. DeepFM

```python
from factorization_rec import DeepFM

# Load and preprocess data
# ...

# Create DeepFM model
deepfm_model = DeepFM()

# Train the model
deepfm_model.fit(train_data)

# Make predictions
predictions = deepfm_model.predict(test_data)
```

### 3. xDeepFM

```python
from factorization_rec import xDeepFM

# Load and preprocess data
# ...

# Create xDeepFM model
xdeepfm_model = xDeepFM()

# Train the model
xdeepfm_model.fit(train_data)

# Make predictions
predictions = xdeepfm_model.predict(test_data)
```

## Contribution

Contributions are welcome! If you find a bug or want to add a new feature, please open an issue or send a pull request.

## License

This library is provided under the MIT license. For more details, refer to the [LICENSE.md](LICENSE.md) file.
