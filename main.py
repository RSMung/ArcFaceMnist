from trainArcFace import trainArcFaceMain
from trainResnet import trainResnetSoftmaxMain


flag1 = "trainArcFaceMain"
flag2 = "trainResnetSoftmaxMain"





flag = flag1
# flag = flag2



if flag == flag1:
    trainArcFaceMain()
elif flag == flag2:
    trainResnetSoftmaxMain()
else:
    raise RuntimeError(f"flag:{flag} is invalid!")