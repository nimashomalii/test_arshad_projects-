from model_use.main import choose_model
import sys
from plot import plot_subject_dependet
emotion = sys.argv[2]
category  = sys.argv[3]
k = int(sys.argv[4])
model_name  = sys.argv[1]
accuraceis = choose_model(model_name ,emotion , category, None , None  , subject_dependecy = 'subject_dependent') 
print(accuraceis['test'])

plot_subject_dependet(accuraceis)

