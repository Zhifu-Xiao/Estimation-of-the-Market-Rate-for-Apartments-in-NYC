import homework2_rent
from sklearn.metrics import r2_score
import numpy as np

def test_rent():
  """ 
  Created a function that checks the R^2 returned by score_rent to be
  as least as good as I expected outcome, which is 0.8.
  """
  X_test, y_test, y_pred = homework2_rent.TestClass().predict_rent()
  test_r2 = r2_score(y_test,y_pred)
  print(test_r2)
  assert(test_r2>=0.8)
