# miml: Multi-Instance Multi-Label Learning Library for Python
The aim of the library is to ease the development, testing, and comparison of classification algorithms for multi-instance multi-label learning (MIML). 

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Usage](#usage)
- [License](#license)

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install miml.

```bash
$ pip install mimllearning
```
#### Requirements
The requirement packages for miml library are: numpy and scikit-learn.
Installing miml with the package manager does not install the package dependencies.
So install them with the package manager manually if not already downloaded.

    $ pip install numpy
    $ pip install scikit-learn

### Documentation

We can find the documentation of the project in this link: [Documentation](https://p82maavd.github.io/MIML/miml)


### Usage


#### Datasets

``` python
from miml.data.load_datasets import load_dataset

dataset_train = load_dataset("miml_birds_random_80train.arff", from_library=True)
dataset_test = load_dataset("C:/Users/Damián/Desktop/miml_birds_random_20test.arff")
```

#### Classifier

``` python
from miml.classifier import MIMLtoMIBRClassifier, AllPositiveAPRClassifier

classifier_mi = MIMLtoMIBRClassifier(AllPositiveAPRClassifier())
classifier_mi.fit(dataset_train)
results_mi=classifier_mi.evaluate(dataset_test)
probs_mi = classifier_mi.predict_proba(dataset_test)
```

#### Report

``` python
from miml.report import Report

report = Report(results_mi, probs_mi, dataset_test)
report.to_string()
print("")
report.to_csv()
```

### License
MIML library is released under the GNU General Public License [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).
