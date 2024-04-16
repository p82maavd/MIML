import mil.metrics
from mil.data.datasets import musk1
from mil.trainer import Trainer
from mil.metrics import *
from mil.validators import LeaveOneOut
from mil.trainer.trainer import Trainer
from mil.models import MILES
from keras.metrics import *
from mil.bag_representation.mapping import MILESMapping
from mil.preprocessing import StandarizerBagsList
from mil import *

(bags_train, y_train), (bags_test, y_test) = musk1.load()

trainer = Trainer()
metrics = [AUC, BinaryAccuracy]
model = MILES()
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', MILESMapping())]

trainer.prepare(model, preprocess_pipeline=pipeline, metrics=metrics)
# fitting trainer

history = trainer.fit(bags_train, y_train, sample_weights='balanced', verbose=1)
