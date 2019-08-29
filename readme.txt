pass an argument to model.py as relative directly to image to be tested as "python model.py ../xxx/20.jpg"
model for graphs are stored in ckpt files
svhn_cnn is used to identify the digit
stage_1_cnn is used to determine whether an image has digit or not using cnn
stage_1_svm is used to determine whether an image has digit or not using svm
char_74k and char_74k_3d represent gray and rgb images of same dataset 74kchar
image_part and image_part_3d are used to partition image into various possible object which proceeds to satge_1_cnn and stage_1_svm which decides whether image has digit or not...which then is passed to svhn_cnn to indentify which digit is there
