
import Testing
import Training
# /////////////////////////////////////////////////////////////////
x = float(input("Enter 1 To train Or 2 To Test : "))
if x == 1:
    train = Training.Training()
    train.training()
else:
    c = Testing.Testing()
    c.testing()

