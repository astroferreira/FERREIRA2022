��.
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.22v2.3.1-38-g9edbe5075f78��(
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
�
ConvBlock-0/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameConvBlock-0/conv2d/kernel
�
-ConvBlock-0/conv2d/kernel/Read/ReadVariableOpReadVariableOpConvBlock-0/conv2d/kernel*&
_output_shapes
: *
dtype0
�
ConvBlock-0/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameConvBlock-0/conv2d/bias

+ConvBlock-0/conv2d/bias/Read/ReadVariableOpReadVariableOpConvBlock-0/conv2d/bias*
_output_shapes
: *
dtype0
�
ConvBlock-0/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameConvBlock-0/conv2d_1/kernel
�
/ConvBlock-0/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpConvBlock-0/conv2d_1/kernel*&
_output_shapes
:  *
dtype0
�
ConvBlock-0/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameConvBlock-0/conv2d_1/bias
�
-ConvBlock-0/conv2d_1/bias/Read/ReadVariableOpReadVariableOpConvBlock-0/conv2d_1/bias*
_output_shapes
: *
dtype0
�
%ConvBlock-0/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%ConvBlock-0/batch_normalization/gamma
�
9ConvBlock-0/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp%ConvBlock-0/batch_normalization/gamma*
_output_shapes
: *
dtype0
�
$ConvBlock-0/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$ConvBlock-0/batch_normalization/beta
�
8ConvBlock-0/batch_normalization/beta/Read/ReadVariableOpReadVariableOp$ConvBlock-0/batch_normalization/beta*
_output_shapes
: *
dtype0
�
'ConvBlock-0/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'ConvBlock-0/batch_normalization_1/gamma
�
;ConvBlock-0/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-0/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
�
&ConvBlock-0/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&ConvBlock-0/batch_normalization_1/beta
�
:ConvBlock-0/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-0/batch_normalization_1/beta*
_output_shapes
: *
dtype0
�
+ConvBlock-0/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+ConvBlock-0/batch_normalization/moving_mean
�
?ConvBlock-0/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp+ConvBlock-0/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
/ConvBlock-0/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/ConvBlock-0/batch_normalization/moving_variance
�
CConvBlock-0/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp/ConvBlock-0/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
�
-ConvBlock-0/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-ConvBlock-0/batch_normalization_1/moving_mean
�
AConvBlock-0/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-0/batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
�
1ConvBlock-0/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31ConvBlock-0/batch_normalization_1/moving_variance
�
EConvBlock-0/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-0/batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
�
ConvBlock-1/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 @*,
shared_nameConvBlock-1/conv2d_2/kernel
�
/ConvBlock-1/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_2/kernel*&
_output_shapes
:		 @*
dtype0
�
ConvBlock-1/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameConvBlock-1/conv2d_2/bias
�
-ConvBlock-1/conv2d_2/bias/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_2/bias*
_output_shapes
:@*
dtype0
�
ConvBlock-1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@@*,
shared_nameConvBlock-1/conv2d_3/kernel
�
/ConvBlock-1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_3/kernel*&
_output_shapes
:		@@*
dtype0
�
ConvBlock-1/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameConvBlock-1/conv2d_3/bias
�
-ConvBlock-1/conv2d_3/bias/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_3/bias*
_output_shapes
:@*
dtype0
�
'ConvBlock-1/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'ConvBlock-1/batch_normalization_2/gamma
�
;ConvBlock-1/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-1/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
&ConvBlock-1/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&ConvBlock-1/batch_normalization_2/beta
�
:ConvBlock-1/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-1/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
�
'ConvBlock-1/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'ConvBlock-1/batch_normalization_3/gamma
�
;ConvBlock-1/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-1/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
�
&ConvBlock-1/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&ConvBlock-1/batch_normalization_3/beta
�
:ConvBlock-1/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-1/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
�
-ConvBlock-1/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-ConvBlock-1/batch_normalization_2/moving_mean
�
AConvBlock-1/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-1/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
�
1ConvBlock-1/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31ConvBlock-1/batch_normalization_2/moving_variance
�
EConvBlock-1/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-1/batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
�
-ConvBlock-1/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-ConvBlock-1/batch_normalization_3/moving_mean
�
AConvBlock-1/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-1/batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
�
1ConvBlock-1/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31ConvBlock-1/batch_normalization_3/moving_variance
�
EConvBlock-1/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-1/batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
�
ConvBlock-2/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*,
shared_nameConvBlock-2/conv2d_4/kernel
�
/ConvBlock-2/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_4/kernel*'
_output_shapes
:@�*
dtype0
�
ConvBlock-2/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameConvBlock-2/conv2d_4/bias
�
-ConvBlock-2/conv2d_4/bias/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_4/bias*
_output_shapes	
:�*
dtype0
�
ConvBlock-2/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*,
shared_nameConvBlock-2/conv2d_5/kernel
�
/ConvBlock-2/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_5/kernel*(
_output_shapes
:��*
dtype0
�
ConvBlock-2/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameConvBlock-2/conv2d_5/bias
�
-ConvBlock-2/conv2d_5/bias/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_5/bias*
_output_shapes	
:�*
dtype0
�
'ConvBlock-2/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'ConvBlock-2/batch_normalization_4/gamma
�
;ConvBlock-2/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-2/batch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
&ConvBlock-2/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&ConvBlock-2/batch_normalization_4/beta
�
:ConvBlock-2/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-2/batch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
'ConvBlock-2/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'ConvBlock-2/batch_normalization_5/gamma
�
;ConvBlock-2/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-2/batch_normalization_5/gamma*
_output_shapes	
:�*
dtype0
�
&ConvBlock-2/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&ConvBlock-2/batch_normalization_5/beta
�
:ConvBlock-2/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-2/batch_normalization_5/beta*
_output_shapes	
:�*
dtype0
�
-ConvBlock-2/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-ConvBlock-2/batch_normalization_4/moving_mean
�
AConvBlock-2/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-2/batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
1ConvBlock-2/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31ConvBlock-2/batch_normalization_4/moving_variance
�
EConvBlock-2/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-2/batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
-ConvBlock-2/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-ConvBlock-2/batch_normalization_5/moving_mean
�
AConvBlock-2/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-2/batch_normalization_5/moving_mean*
_output_shapes	
:�*
dtype0
�
1ConvBlock-2/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31ConvBlock-2/batch_normalization_5/moving_variance
�
EConvBlock-2/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-2/batch_normalization_5/moving_variance*
_output_shapes	
:�*
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
l
total_cmVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
total_cm
e
total_cm/Read/ReadVariableOpReadVariableOptotal_cm*
_output_shapes

:*
dtype0
�
SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��**
shared_nameSGD/dense/kernel/momentum
�
-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*(
shared_nameSGD/dense/bias/momentum
�
+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGD/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*,
shared_nameSGD/dense_1/kernel/momentum
�
/SGD/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
SGD/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameSGD/dense_1/bias/momentum
�
-SGD/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGD/dense_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_nameSGD/dense_2/kernel/momentum
�
/SGD/dense_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/kernel/momentum*
_output_shapes
:	�*
dtype0
�
SGD/dense_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_2/bias/momentum
�
-SGD/dense_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/bias/momentum*
_output_shapes
:*
dtype0
�
&SGD/ConvBlock-0/conv2d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&SGD/ConvBlock-0/conv2d/kernel/momentum
�
:SGD/ConvBlock-0/conv2d/kernel/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-0/conv2d/kernel/momentum*&
_output_shapes
: *
dtype0
�
$SGD/ConvBlock-0/conv2d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$SGD/ConvBlock-0/conv2d/bias/momentum
�
8SGD/ConvBlock-0/conv2d/bias/momentum/Read/ReadVariableOpReadVariableOp$SGD/ConvBlock-0/conv2d/bias/momentum*
_output_shapes
: *
dtype0
�
(SGD/ConvBlock-0/conv2d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *9
shared_name*(SGD/ConvBlock-0/conv2d_1/kernel/momentum
�
<SGD/ConvBlock-0/conv2d_1/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-0/conv2d_1/kernel/momentum*&
_output_shapes
:  *
dtype0
�
&SGD/ConvBlock-0/conv2d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&SGD/ConvBlock-0/conv2d_1/bias/momentum
�
:SGD/ConvBlock-0/conv2d_1/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-0/conv2d_1/bias/momentum*
_output_shapes
: *
dtype0
�
2SGD/ConvBlock-0/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42SGD/ConvBlock-0/batch_normalization/gamma/momentum
�
FSGD/ConvBlock-0/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp2SGD/ConvBlock-0/batch_normalization/gamma/momentum*
_output_shapes
: *
dtype0
�
1SGD/ConvBlock-0/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31SGD/ConvBlock-0/batch_normalization/beta/momentum
�
ESGD/ConvBlock-0/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp1SGD/ConvBlock-0/batch_normalization/beta/momentum*
_output_shapes
: *
dtype0
�
4SGD/ConvBlock-0/batch_normalization_1/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64SGD/ConvBlock-0/batch_normalization_1/gamma/momentum
�
HSGD/ConvBlock-0/batch_normalization_1/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-0/batch_normalization_1/gamma/momentum*
_output_shapes
: *
dtype0
�
3SGD/ConvBlock-0/batch_normalization_1/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53SGD/ConvBlock-0/batch_normalization_1/beta/momentum
�
GSGD/ConvBlock-0/batch_normalization_1/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-0/batch_normalization_1/beta/momentum*
_output_shapes
: *
dtype0
�
(SGD/ConvBlock-1/conv2d_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 @*9
shared_name*(SGD/ConvBlock-1/conv2d_2/kernel/momentum
�
<SGD/ConvBlock-1/conv2d_2/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-1/conv2d_2/kernel/momentum*&
_output_shapes
:		 @*
dtype0
�
&SGD/ConvBlock-1/conv2d_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&SGD/ConvBlock-1/conv2d_2/bias/momentum
�
:SGD/ConvBlock-1/conv2d_2/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-1/conv2d_2/bias/momentum*
_output_shapes
:@*
dtype0
�
(SGD/ConvBlock-1/conv2d_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@@*9
shared_name*(SGD/ConvBlock-1/conv2d_3/kernel/momentum
�
<SGD/ConvBlock-1/conv2d_3/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-1/conv2d_3/kernel/momentum*&
_output_shapes
:		@@*
dtype0
�
&SGD/ConvBlock-1/conv2d_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&SGD/ConvBlock-1/conv2d_3/bias/momentum
�
:SGD/ConvBlock-1/conv2d_3/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-1/conv2d_3/bias/momentum*
_output_shapes
:@*
dtype0
�
4SGD/ConvBlock-1/batch_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64SGD/ConvBlock-1/batch_normalization_2/gamma/momentum
�
HSGD/ConvBlock-1/batch_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-1/batch_normalization_2/gamma/momentum*
_output_shapes
:@*
dtype0
�
3SGD/ConvBlock-1/batch_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53SGD/ConvBlock-1/batch_normalization_2/beta/momentum
�
GSGD/ConvBlock-1/batch_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-1/batch_normalization_2/beta/momentum*
_output_shapes
:@*
dtype0
�
4SGD/ConvBlock-1/batch_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64SGD/ConvBlock-1/batch_normalization_3/gamma/momentum
�
HSGD/ConvBlock-1/batch_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-1/batch_normalization_3/gamma/momentum*
_output_shapes
:@*
dtype0
�
3SGD/ConvBlock-1/batch_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53SGD/ConvBlock-1/batch_normalization_3/beta/momentum
�
GSGD/ConvBlock-1/batch_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-1/batch_normalization_3/beta/momentum*
_output_shapes
:@*
dtype0
�
(SGD/ConvBlock-2/conv2d_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*9
shared_name*(SGD/ConvBlock-2/conv2d_4/kernel/momentum
�
<SGD/ConvBlock-2/conv2d_4/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-2/conv2d_4/kernel/momentum*'
_output_shapes
:@�*
dtype0
�
&SGD/ConvBlock-2/conv2d_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&SGD/ConvBlock-2/conv2d_4/bias/momentum
�
:SGD/ConvBlock-2/conv2d_4/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-2/conv2d_4/bias/momentum*
_output_shapes	
:�*
dtype0
�
(SGD/ConvBlock-2/conv2d_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*9
shared_name*(SGD/ConvBlock-2/conv2d_5/kernel/momentum
�
<SGD/ConvBlock-2/conv2d_5/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-2/conv2d_5/kernel/momentum*(
_output_shapes
:��*
dtype0
�
&SGD/ConvBlock-2/conv2d_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&SGD/ConvBlock-2/conv2d_5/bias/momentum
�
:SGD/ConvBlock-2/conv2d_5/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-2/conv2d_5/bias/momentum*
_output_shapes	
:�*
dtype0
�
4SGD/ConvBlock-2/batch_normalization_4/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64SGD/ConvBlock-2/batch_normalization_4/gamma/momentum
�
HSGD/ConvBlock-2/batch_normalization_4/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-2/batch_normalization_4/gamma/momentum*
_output_shapes	
:�*
dtype0
�
3SGD/ConvBlock-2/batch_normalization_4/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53SGD/ConvBlock-2/batch_normalization_4/beta/momentum
�
GSGD/ConvBlock-2/batch_normalization_4/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-2/batch_normalization_4/beta/momentum*
_output_shapes	
:�*
dtype0
�
4SGD/ConvBlock-2/batch_normalization_5/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64SGD/ConvBlock-2/batch_normalization_5/gamma/momentum
�
HSGD/ConvBlock-2/batch_normalization_5/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-2/batch_normalization_5/gamma/momentum*
_output_shapes	
:�*
dtype0
�
3SGD/ConvBlock-2/batch_normalization_5/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53SGD/ConvBlock-2/batch_normalization_5/beta/momentum
�
GSGD/ConvBlock-2/batch_normalization_5/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-2/batch_normalization_5/beta/momentum*
_output_shapes	
:�*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
\
_rng
regularization_losses
	variables
trainable_variables
	keras_api
~

conv2d

activation

batch_norm
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
~

&conv2d
'
activation
(
batch_norm
)regularization_losses
*	variables
+trainable_variables
,	keras_api
R
-regularization_losses
.	variables
/trainable_variables
0	keras_api
~

1conv2d
2
activation
3
batch_norm
4regularization_losses
5	variables
6trainable_variables
7	keras_api
R
8regularization_losses
9	variables
:trainable_variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
R
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
R
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
h

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
�
	^decay
_learning_rate
`momentum
aiter<momentum�=momentum�Jmomentum�Kmomentum�Xmomentum�Ymomentum�bmomentum�cmomentum�dmomentum�emomentum�fmomentum�gmomentum�hmomentum�imomentum�nmomentum�omomentum�pmomentum�qmomentum�rmomentum�smomentum�tmomentum�umomentum�zmomentum�{momentum�|momentum�}momentum�~momentum�momentum��momentum��momentum�
 
�
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11
n12
o13
p14
q15
r16
s17
t18
u19
v20
w21
x22
y23
z24
{25
|26
}27
~28
29
�30
�31
�32
�33
�34
�35
<36
=37
J38
K39
X40
Y41
�
b0
c1
d2
e3
f4
g5
h6
i7
n8
o9
p10
q11
r12
s13
t14
u15
z16
{17
|18
}19
~20
21
�22
�23
<24
=25
J26
K27
X28
Y29
�
�layer_metrics
regularization_losses
�non_trainable_variables
�metrics
	variables
�layers
trainable_variables
 �layer_regularization_losses
 

�
_state_var
 
 
 
�
�layer_metrics
�non_trainable_variables
regularization_losses
�metrics
	variables
�layers
trainable_variables
 �layer_regularization_losses

�0
�1

�0
�1

�0
�1
 
V
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11
8
b0
c1
d2
e3
f4
g5
h6
i7
�
�layer_metrics
�non_trainable_variables
regularization_losses
�metrics
	variables
�layers
 trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
"regularization_losses
�metrics
#	variables
�layers
$trainable_variables
 �layer_regularization_losses

�0
�1

�0
�1

�0
�1
 
V
n0
o1
p2
q3
r4
s5
t6
u7
v8
w9
x10
y11
8
n0
o1
p2
q3
r4
s5
t6
u7
�
�layer_metrics
�non_trainable_variables
)regularization_losses
�metrics
*	variables
�layers
+trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
-regularization_losses
�metrics
.	variables
�layers
/trainable_variables
 �layer_regularization_losses

�0
�1

�0
�1

�0
�1
 
\
z0
{1
|2
}3
~4
5
�6
�7
�8
�9
�10
�11
:
z0
{1
|2
}3
~4
5
�6
�7
�
�layer_metrics
�non_trainable_variables
4regularization_losses
�metrics
5	variables
�layers
6trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
8regularization_losses
�metrics
9	variables
�layers
:trainable_variables
 �layer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
�
�layer_metrics
�non_trainable_variables
>regularization_losses
�metrics
?	variables
�layers
@trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
Bregularization_losses
�metrics
C	variables
�layers
Dtrainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
Fregularization_losses
�metrics
G	variables
�layers
Htrainable_variables
 �layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
�
�layer_metrics
�non_trainable_variables
Lregularization_losses
�metrics
M	variables
�layers
Ntrainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
Pregularization_losses
�metrics
Q	variables
�layers
Rtrainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
Tregularization_losses
�metrics
U	variables
�layers
Vtrainable_variables
 �layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
�
�layer_metrics
�non_trainable_variables
Zregularization_losses
�metrics
[	variables
�layers
\trainable_variables
 �layer_regularization_losses
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEConvBlock-0/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEConvBlock-0/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConvBlock-0/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEConvBlock-0/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%ConvBlock-0/batch_normalization/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$ConvBlock-0/batch_normalization/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'ConvBlock-0/batch_normalization_1/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&ConvBlock-0/batch_normalization_1/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+ConvBlock-0/batch_normalization/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE/ConvBlock-0/batch_normalization/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-ConvBlock-0/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1ConvBlock-0/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEConvBlock-1/conv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEConvBlock-1/conv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEConvBlock-1/conv2d_3/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEConvBlock-1/conv2d_3/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'ConvBlock-1/batch_normalization_2/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&ConvBlock-1/batch_normalization_2/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'ConvBlock-1/batch_normalization_3/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&ConvBlock-1/batch_normalization_3/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-ConvBlock-1/batch_normalization_2/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1ConvBlock-1/batch_normalization_2/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-ConvBlock-1/batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1ConvBlock-1/batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEConvBlock-2/conv2d_4/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEConvBlock-2/conv2d_4/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEConvBlock-2/conv2d_5/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEConvBlock-2/conv2d_5/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'ConvBlock-2/batch_normalization_4/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&ConvBlock-2/batch_normalization_4/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'ConvBlock-2/batch_normalization_5/gamma'variables/30/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&ConvBlock-2/batch_normalization_5/beta'variables/31/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-ConvBlock-2/batch_normalization_4/moving_mean'variables/32/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1ConvBlock-2/batch_normalization_4/moving_variance'variables/33/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-ConvBlock-2/batch_normalization_5/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1ConvBlock-2/batch_normalization_5/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
 
Z
j0
k1
l2
m3
v4
w5
x6
y7
�8
�9
�10
�11
(
�0
�1
�2
�3
�4
n
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
 
PN
VARIABLE_VALUEVariable2layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
l

bkernel
cbias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

dkernel
ebias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
	�axis
	fgamma
gbeta
jmoving_mean
kmoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
	�axis
	hgamma
ibeta
lmoving_mean
mmoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 

j0
k1
l2
m3
 
0
�0
�1
�2
�3
�4
�5
 
 
 
 
 
 
l

nkernel
obias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

pkernel
qbias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
	�axis
	rgamma
sbeta
vmoving_mean
wmoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
	�axis
	tgamma
ubeta
xmoving_mean
ymoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 

v0
w1
x2
y3
 
0
�0
�1
�2
�3
�4
�5
 
 
 
 
 
 
l

zkernel
{bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

|kernel
}bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
	�axis
	~gamma
beta
�moving_mean
�moving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 
 
�0
�1
�2
�3
 
0
�0
�1
�2
�3
�4
�5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
\
�
thresholds
�true_positives
�false_positives
�	variables
�	keras_api
\
�
thresholds
�true_positives
�false_negatives
�	variables
�	keras_api
/
�total_cm
�	variables
�	keras_api
 

b0
c1

b0
c1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 

d0
e1

d0
e1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 

f0
g1
j2
k3

f0
g1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 

h0
i1
l2
m3

h0
i1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 

n0
o1

n0
o1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 

p0
q1

p0
q1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 

r0
s1
v2
w3

r0
s1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 

t0
u1
x2
y3

t0
u1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 

z0
{1

z0
{1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 

|0
}1

|0
}1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 

~0
1
�2
�3

~0
1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�0
�1
�2
�3

�0
�1
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
US
VARIABLE_VALUEtotal_cm7keras_api/metrics/4/total_cm/.ATTRIBUTES/VARIABLE_VALUE

�0

�	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

j0
k1
 
 
 
 

l0
m1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

v0
w1
 
 
 
 

x0
y1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 

�0
�1
 
 
 
��
VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_1/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_1/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_2/kernel/momentumYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_2/bias/momentumWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&SGD/ConvBlock-0/conv2d/kernel/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE$SGD/ConvBlock-0/conv2d/bias/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/ConvBlock-0/conv2d_1/kernel/momentumIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&SGD/ConvBlock-0/conv2d_1/bias/momentumIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2SGD/ConvBlock-0/batch_normalization/gamma/momentumIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE1SGD/ConvBlock-0/batch_normalization/beta/momentumIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4SGD/ConvBlock-0/batch_normalization_1/gamma/momentumIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3SGD/ConvBlock-0/batch_normalization_1/beta/momentumIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/ConvBlock-1/conv2d_2/kernel/momentumJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&SGD/ConvBlock-1/conv2d_2/bias/momentumJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/ConvBlock-1/conv2d_3/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&SGD/ConvBlock-1/conv2d_3/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4SGD/ConvBlock-1/batch_normalization_2/gamma/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3SGD/ConvBlock-1/batch_normalization_2/beta/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4SGD/ConvBlock-1/batch_normalization_3/gamma/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3SGD/ConvBlock-1/batch_normalization_3/beta/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/ConvBlock-2/conv2d_4/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&SGD/ConvBlock-2/conv2d_4/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/ConvBlock-2/conv2d_5/kernel/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&SGD/ConvBlock-2/conv2d_5/bias/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4SGD/ConvBlock-2/batch_normalization_4/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3SGD/ConvBlock-2/batch_normalization_4/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4SGD/ConvBlock-2/batch_normalization_5/gamma/momentumJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3SGD/ConvBlock-2/batch_normalization_5/beta/momentumJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1ConvBlock-0/conv2d/kernelConvBlock-0/conv2d/bias%ConvBlock-0/batch_normalization/gamma$ConvBlock-0/batch_normalization/beta+ConvBlock-0/batch_normalization/moving_mean/ConvBlock-0/batch_normalization/moving_varianceConvBlock-0/conv2d_1/kernelConvBlock-0/conv2d_1/bias'ConvBlock-0/batch_normalization_1/gamma&ConvBlock-0/batch_normalization_1/beta-ConvBlock-0/batch_normalization_1/moving_mean1ConvBlock-0/batch_normalization_1/moving_varianceConvBlock-1/conv2d_2/kernelConvBlock-1/conv2d_2/bias'ConvBlock-1/batch_normalization_2/gamma&ConvBlock-1/batch_normalization_2/beta-ConvBlock-1/batch_normalization_2/moving_mean1ConvBlock-1/batch_normalization_2/moving_varianceConvBlock-1/conv2d_3/kernelConvBlock-1/conv2d_3/bias'ConvBlock-1/batch_normalization_3/gamma&ConvBlock-1/batch_normalization_3/beta-ConvBlock-1/batch_normalization_3/moving_mean1ConvBlock-1/batch_normalization_3/moving_varianceConvBlock-2/conv2d_4/kernelConvBlock-2/conv2d_4/bias'ConvBlock-2/batch_normalization_4/gamma&ConvBlock-2/batch_normalization_4/beta-ConvBlock-2/batch_normalization_4/moving_mean1ConvBlock-2/batch_normalization_4/moving_varianceConvBlock-2/conv2d_5/kernelConvBlock-2/conv2d_5/bias'ConvBlock-2/batch_normalization_5/gamma&ConvBlock-2/batch_normalization_5/beta-ConvBlock-2/batch_normalization_5/moving_mean1ConvBlock-2/batch_normalization_5/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_414353
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOp-ConvBlock-0/conv2d/kernel/Read/ReadVariableOp+ConvBlock-0/conv2d/bias/Read/ReadVariableOp/ConvBlock-0/conv2d_1/kernel/Read/ReadVariableOp-ConvBlock-0/conv2d_1/bias/Read/ReadVariableOp9ConvBlock-0/batch_normalization/gamma/Read/ReadVariableOp8ConvBlock-0/batch_normalization/beta/Read/ReadVariableOp;ConvBlock-0/batch_normalization_1/gamma/Read/ReadVariableOp:ConvBlock-0/batch_normalization_1/beta/Read/ReadVariableOp?ConvBlock-0/batch_normalization/moving_mean/Read/ReadVariableOpCConvBlock-0/batch_normalization/moving_variance/Read/ReadVariableOpAConvBlock-0/batch_normalization_1/moving_mean/Read/ReadVariableOpEConvBlock-0/batch_normalization_1/moving_variance/Read/ReadVariableOp/ConvBlock-1/conv2d_2/kernel/Read/ReadVariableOp-ConvBlock-1/conv2d_2/bias/Read/ReadVariableOp/ConvBlock-1/conv2d_3/kernel/Read/ReadVariableOp-ConvBlock-1/conv2d_3/bias/Read/ReadVariableOp;ConvBlock-1/batch_normalization_2/gamma/Read/ReadVariableOp:ConvBlock-1/batch_normalization_2/beta/Read/ReadVariableOp;ConvBlock-1/batch_normalization_3/gamma/Read/ReadVariableOp:ConvBlock-1/batch_normalization_3/beta/Read/ReadVariableOpAConvBlock-1/batch_normalization_2/moving_mean/Read/ReadVariableOpEConvBlock-1/batch_normalization_2/moving_variance/Read/ReadVariableOpAConvBlock-1/batch_normalization_3/moving_mean/Read/ReadVariableOpEConvBlock-1/batch_normalization_3/moving_variance/Read/ReadVariableOp/ConvBlock-2/conv2d_4/kernel/Read/ReadVariableOp-ConvBlock-2/conv2d_4/bias/Read/ReadVariableOp/ConvBlock-2/conv2d_5/kernel/Read/ReadVariableOp-ConvBlock-2/conv2d_5/bias/Read/ReadVariableOp;ConvBlock-2/batch_normalization_4/gamma/Read/ReadVariableOp:ConvBlock-2/batch_normalization_4/beta/Read/ReadVariableOp;ConvBlock-2/batch_normalization_5/gamma/Read/ReadVariableOp:ConvBlock-2/batch_normalization_5/beta/Read/ReadVariableOpAConvBlock-2/batch_normalization_4/moving_mean/Read/ReadVariableOpEConvBlock-2/batch_normalization_4/moving_variance/Read/ReadVariableOpAConvBlock-2/batch_normalization_5/moving_mean/Read/ReadVariableOpEConvBlock-2/batch_normalization_5/moving_variance/Read/ReadVariableOpVariable/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOptotal_cm/Read/ReadVariableOp-SGD/dense/kernel/momentum/Read/ReadVariableOp+SGD/dense/bias/momentum/Read/ReadVariableOp/SGD/dense_1/kernel/momentum/Read/ReadVariableOp-SGD/dense_1/bias/momentum/Read/ReadVariableOp/SGD/dense_2/kernel/momentum/Read/ReadVariableOp-SGD/dense_2/bias/momentum/Read/ReadVariableOp:SGD/ConvBlock-0/conv2d/kernel/momentum/Read/ReadVariableOp8SGD/ConvBlock-0/conv2d/bias/momentum/Read/ReadVariableOp<SGD/ConvBlock-0/conv2d_1/kernel/momentum/Read/ReadVariableOp:SGD/ConvBlock-0/conv2d_1/bias/momentum/Read/ReadVariableOpFSGD/ConvBlock-0/batch_normalization/gamma/momentum/Read/ReadVariableOpESGD/ConvBlock-0/batch_normalization/beta/momentum/Read/ReadVariableOpHSGD/ConvBlock-0/batch_normalization_1/gamma/momentum/Read/ReadVariableOpGSGD/ConvBlock-0/batch_normalization_1/beta/momentum/Read/ReadVariableOp<SGD/ConvBlock-1/conv2d_2/kernel/momentum/Read/ReadVariableOp:SGD/ConvBlock-1/conv2d_2/bias/momentum/Read/ReadVariableOp<SGD/ConvBlock-1/conv2d_3/kernel/momentum/Read/ReadVariableOp:SGD/ConvBlock-1/conv2d_3/bias/momentum/Read/ReadVariableOpHSGD/ConvBlock-1/batch_normalization_2/gamma/momentum/Read/ReadVariableOpGSGD/ConvBlock-1/batch_normalization_2/beta/momentum/Read/ReadVariableOpHSGD/ConvBlock-1/batch_normalization_3/gamma/momentum/Read/ReadVariableOpGSGD/ConvBlock-1/batch_normalization_3/beta/momentum/Read/ReadVariableOp<SGD/ConvBlock-2/conv2d_4/kernel/momentum/Read/ReadVariableOp:SGD/ConvBlock-2/conv2d_4/bias/momentum/Read/ReadVariableOp<SGD/ConvBlock-2/conv2d_5/kernel/momentum/Read/ReadVariableOp:SGD/ConvBlock-2/conv2d_5/bias/momentum/Read/ReadVariableOpHSGD/ConvBlock-2/batch_normalization_4/gamma/momentum/Read/ReadVariableOpGSGD/ConvBlock-2/batch_normalization_4/beta/momentum/Read/ReadVariableOpHSGD/ConvBlock-2/batch_normalization_5/gamma/momentum/Read/ReadVariableOpGSGD/ConvBlock-2/batch_normalization_5/beta/momentum/Read/ReadVariableOpConst*c
Tin\
Z2X		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_416919
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdecaylearning_ratemomentumSGD/iterConvBlock-0/conv2d/kernelConvBlock-0/conv2d/biasConvBlock-0/conv2d_1/kernelConvBlock-0/conv2d_1/bias%ConvBlock-0/batch_normalization/gamma$ConvBlock-0/batch_normalization/beta'ConvBlock-0/batch_normalization_1/gamma&ConvBlock-0/batch_normalization_1/beta+ConvBlock-0/batch_normalization/moving_mean/ConvBlock-0/batch_normalization/moving_variance-ConvBlock-0/batch_normalization_1/moving_mean1ConvBlock-0/batch_normalization_1/moving_varianceConvBlock-1/conv2d_2/kernelConvBlock-1/conv2d_2/biasConvBlock-1/conv2d_3/kernelConvBlock-1/conv2d_3/bias'ConvBlock-1/batch_normalization_2/gamma&ConvBlock-1/batch_normalization_2/beta'ConvBlock-1/batch_normalization_3/gamma&ConvBlock-1/batch_normalization_3/beta-ConvBlock-1/batch_normalization_2/moving_mean1ConvBlock-1/batch_normalization_2/moving_variance-ConvBlock-1/batch_normalization_3/moving_mean1ConvBlock-1/batch_normalization_3/moving_varianceConvBlock-2/conv2d_4/kernelConvBlock-2/conv2d_4/biasConvBlock-2/conv2d_5/kernelConvBlock-2/conv2d_5/bias'ConvBlock-2/batch_normalization_4/gamma&ConvBlock-2/batch_normalization_4/beta'ConvBlock-2/batch_normalization_5/gamma&ConvBlock-2/batch_normalization_5/beta-ConvBlock-2/batch_normalization_4/moving_mean1ConvBlock-2/batch_normalization_4/moving_variance-ConvBlock-2/batch_normalization_5/moving_mean1ConvBlock-2/batch_normalization_5/moving_varianceVariabletotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativestotal_cmSGD/dense/kernel/momentumSGD/dense/bias/momentumSGD/dense_1/kernel/momentumSGD/dense_1/bias/momentumSGD/dense_2/kernel/momentumSGD/dense_2/bias/momentum&SGD/ConvBlock-0/conv2d/kernel/momentum$SGD/ConvBlock-0/conv2d/bias/momentum(SGD/ConvBlock-0/conv2d_1/kernel/momentum&SGD/ConvBlock-0/conv2d_1/bias/momentum2SGD/ConvBlock-0/batch_normalization/gamma/momentum1SGD/ConvBlock-0/batch_normalization/beta/momentum4SGD/ConvBlock-0/batch_normalization_1/gamma/momentum3SGD/ConvBlock-0/batch_normalization_1/beta/momentum(SGD/ConvBlock-1/conv2d_2/kernel/momentum&SGD/ConvBlock-1/conv2d_2/bias/momentum(SGD/ConvBlock-1/conv2d_3/kernel/momentum&SGD/ConvBlock-1/conv2d_3/bias/momentum4SGD/ConvBlock-1/batch_normalization_2/gamma/momentum3SGD/ConvBlock-1/batch_normalization_2/beta/momentum4SGD/ConvBlock-1/batch_normalization_3/gamma/momentum3SGD/ConvBlock-1/batch_normalization_3/beta/momentum(SGD/ConvBlock-2/conv2d_4/kernel/momentum&SGD/ConvBlock-2/conv2d_4/bias/momentum(SGD/ConvBlock-2/conv2d_5/kernel/momentum&SGD/ConvBlock-2/conv2d_5/bias/momentum4SGD/ConvBlock-2/batch_normalization_4/gamma/momentum3SGD/ConvBlock-2/batch_normalization_4/beta/momentum4SGD/ConvBlock-2/batch_normalization_5/gamma/momentum3SGD/ConvBlock-2/batch_normalization_5/beta/momentum*b
Tin[
Y2W*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_417187��%
�
o
6__inference_monte_carlo_dropout_1_layer_call_fn_416015

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_4133582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
3__inference_FERREIRA2020_class_layer_call_fn_415133

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_4140972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
,__inference_ConvBlock-2_layer_call_fn_415933
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_4131752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������@

_user_specified_namex
�
I
-__inference_activation_7_layer_call_fn_416025

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_4133712
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_416594

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_411993

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
I
-__inference_activation_6_layer_call_fn_415979

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_4133122
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_416409

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4120612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
y
__inference_loss_fn_7_416138H
Dconvblock_1_conv2d_3_bias_regularizer_square_readvariableop_resource
identity��
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_1_conv2d_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mulp
IdentityIdentity-ConvBlock-1/conv2d_3/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_412165

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_411872

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
U
9__inference_global_average_pooling2d_layer_call_fn_412440

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4124342
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_416396

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�S
�
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_415875
x+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity��
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_4/BiasAdd�
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_4/Relu�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_5/Conv2D�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_5/BiasAdd�
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_5/Relu�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_5/ReadVariableOp_1�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������@:::::::::::::R N
/
_output_shapes
:���������@

_user_specified_namex
�
y
__inference_loss_fn_5_416116H
Dconvblock_1_conv2d_2_bias_regularizer_square_readvariableop_resource
identity��
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_1_conv2d_2_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulp
IdentityIdentity-ConvBlock-1/conv2d_2/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
��
�,
__inference__traced_save_416919
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	8
4savev2_convblock_0_conv2d_kernel_read_readvariableop6
2savev2_convblock_0_conv2d_bias_read_readvariableop:
6savev2_convblock_0_conv2d_1_kernel_read_readvariableop8
4savev2_convblock_0_conv2d_1_bias_read_readvariableopD
@savev2_convblock_0_batch_normalization_gamma_read_readvariableopC
?savev2_convblock_0_batch_normalization_beta_read_readvariableopF
Bsavev2_convblock_0_batch_normalization_1_gamma_read_readvariableopE
Asavev2_convblock_0_batch_normalization_1_beta_read_readvariableopJ
Fsavev2_convblock_0_batch_normalization_moving_mean_read_readvariableopN
Jsavev2_convblock_0_batch_normalization_moving_variance_read_readvariableopL
Hsavev2_convblock_0_batch_normalization_1_moving_mean_read_readvariableopP
Lsavev2_convblock_0_batch_normalization_1_moving_variance_read_readvariableop:
6savev2_convblock_1_conv2d_2_kernel_read_readvariableop8
4savev2_convblock_1_conv2d_2_bias_read_readvariableop:
6savev2_convblock_1_conv2d_3_kernel_read_readvariableop8
4savev2_convblock_1_conv2d_3_bias_read_readvariableopF
Bsavev2_convblock_1_batch_normalization_2_gamma_read_readvariableopE
Asavev2_convblock_1_batch_normalization_2_beta_read_readvariableopF
Bsavev2_convblock_1_batch_normalization_3_gamma_read_readvariableopE
Asavev2_convblock_1_batch_normalization_3_beta_read_readvariableopL
Hsavev2_convblock_1_batch_normalization_2_moving_mean_read_readvariableopP
Lsavev2_convblock_1_batch_normalization_2_moving_variance_read_readvariableopL
Hsavev2_convblock_1_batch_normalization_3_moving_mean_read_readvariableopP
Lsavev2_convblock_1_batch_normalization_3_moving_variance_read_readvariableop:
6savev2_convblock_2_conv2d_4_kernel_read_readvariableop8
4savev2_convblock_2_conv2d_4_bias_read_readvariableop:
6savev2_convblock_2_conv2d_5_kernel_read_readvariableop8
4savev2_convblock_2_conv2d_5_bias_read_readvariableopF
Bsavev2_convblock_2_batch_normalization_4_gamma_read_readvariableopE
Asavev2_convblock_2_batch_normalization_4_beta_read_readvariableopF
Bsavev2_convblock_2_batch_normalization_5_gamma_read_readvariableopE
Asavev2_convblock_2_batch_normalization_5_beta_read_readvariableopL
Hsavev2_convblock_2_batch_normalization_4_moving_mean_read_readvariableopP
Lsavev2_convblock_2_batch_normalization_4_moving_variance_read_readvariableopL
Hsavev2_convblock_2_batch_normalization_5_moving_mean_read_readvariableopP
Lsavev2_convblock_2_batch_normalization_5_moving_variance_read_readvariableop'
#savev2_variable_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop'
#savev2_total_cm_read_readvariableop8
4savev2_sgd_dense_kernel_momentum_read_readvariableop6
2savev2_sgd_dense_bias_momentum_read_readvariableop:
6savev2_sgd_dense_1_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_1_bias_momentum_read_readvariableop:
6savev2_sgd_dense_2_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_2_bias_momentum_read_readvariableopE
Asavev2_sgd_convblock_0_conv2d_kernel_momentum_read_readvariableopC
?savev2_sgd_convblock_0_conv2d_bias_momentum_read_readvariableopG
Csavev2_sgd_convblock_0_conv2d_1_kernel_momentum_read_readvariableopE
Asavev2_sgd_convblock_0_conv2d_1_bias_momentum_read_readvariableopQ
Msavev2_sgd_convblock_0_batch_normalization_gamma_momentum_read_readvariableopP
Lsavev2_sgd_convblock_0_batch_normalization_beta_momentum_read_readvariableopS
Osavev2_sgd_convblock_0_batch_normalization_1_gamma_momentum_read_readvariableopR
Nsavev2_sgd_convblock_0_batch_normalization_1_beta_momentum_read_readvariableopG
Csavev2_sgd_convblock_1_conv2d_2_kernel_momentum_read_readvariableopE
Asavev2_sgd_convblock_1_conv2d_2_bias_momentum_read_readvariableopG
Csavev2_sgd_convblock_1_conv2d_3_kernel_momentum_read_readvariableopE
Asavev2_sgd_convblock_1_conv2d_3_bias_momentum_read_readvariableopS
Osavev2_sgd_convblock_1_batch_normalization_2_gamma_momentum_read_readvariableopR
Nsavev2_sgd_convblock_1_batch_normalization_2_beta_momentum_read_readvariableopS
Osavev2_sgd_convblock_1_batch_normalization_3_gamma_momentum_read_readvariableopR
Nsavev2_sgd_convblock_1_batch_normalization_3_beta_momentum_read_readvariableopG
Csavev2_sgd_convblock_2_conv2d_4_kernel_momentum_read_readvariableopE
Asavev2_sgd_convblock_2_conv2d_4_bias_momentum_read_readvariableopG
Csavev2_sgd_convblock_2_conv2d_5_kernel_momentum_read_readvariableopE
Asavev2_sgd_convblock_2_conv2d_5_bias_momentum_read_readvariableopS
Osavev2_sgd_convblock_2_batch_normalization_4_gamma_momentum_read_readvariableopR
Nsavev2_sgd_convblock_2_batch_normalization_4_beta_momentum_read_readvariableopS
Osavev2_sgd_convblock_2_batch_normalization_5_gamma_momentum_read_readvariableopR
Nsavev2_sgd_convblock_2_batch_normalization_5_beta_momentum_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_07f0801164324dc6bf1dc4b4c71ff19f/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*�&
value�&B�&WB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB2layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB7keras_api/metrics/4/total_cm/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*�
value�B�WB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop4savev2_convblock_0_conv2d_kernel_read_readvariableop2savev2_convblock_0_conv2d_bias_read_readvariableop6savev2_convblock_0_conv2d_1_kernel_read_readvariableop4savev2_convblock_0_conv2d_1_bias_read_readvariableop@savev2_convblock_0_batch_normalization_gamma_read_readvariableop?savev2_convblock_0_batch_normalization_beta_read_readvariableopBsavev2_convblock_0_batch_normalization_1_gamma_read_readvariableopAsavev2_convblock_0_batch_normalization_1_beta_read_readvariableopFsavev2_convblock_0_batch_normalization_moving_mean_read_readvariableopJsavev2_convblock_0_batch_normalization_moving_variance_read_readvariableopHsavev2_convblock_0_batch_normalization_1_moving_mean_read_readvariableopLsavev2_convblock_0_batch_normalization_1_moving_variance_read_readvariableop6savev2_convblock_1_conv2d_2_kernel_read_readvariableop4savev2_convblock_1_conv2d_2_bias_read_readvariableop6savev2_convblock_1_conv2d_3_kernel_read_readvariableop4savev2_convblock_1_conv2d_3_bias_read_readvariableopBsavev2_convblock_1_batch_normalization_2_gamma_read_readvariableopAsavev2_convblock_1_batch_normalization_2_beta_read_readvariableopBsavev2_convblock_1_batch_normalization_3_gamma_read_readvariableopAsavev2_convblock_1_batch_normalization_3_beta_read_readvariableopHsavev2_convblock_1_batch_normalization_2_moving_mean_read_readvariableopLsavev2_convblock_1_batch_normalization_2_moving_variance_read_readvariableopHsavev2_convblock_1_batch_normalization_3_moving_mean_read_readvariableopLsavev2_convblock_1_batch_normalization_3_moving_variance_read_readvariableop6savev2_convblock_2_conv2d_4_kernel_read_readvariableop4savev2_convblock_2_conv2d_4_bias_read_readvariableop6savev2_convblock_2_conv2d_5_kernel_read_readvariableop4savev2_convblock_2_conv2d_5_bias_read_readvariableopBsavev2_convblock_2_batch_normalization_4_gamma_read_readvariableopAsavev2_convblock_2_batch_normalization_4_beta_read_readvariableopBsavev2_convblock_2_batch_normalization_5_gamma_read_readvariableopAsavev2_convblock_2_batch_normalization_5_beta_read_readvariableopHsavev2_convblock_2_batch_normalization_4_moving_mean_read_readvariableopLsavev2_convblock_2_batch_normalization_4_moving_variance_read_readvariableopHsavev2_convblock_2_batch_normalization_5_moving_mean_read_readvariableopLsavev2_convblock_2_batch_normalization_5_moving_variance_read_readvariableop#savev2_variable_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop#savev2_total_cm_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableop6savev2_sgd_dense_1_kernel_momentum_read_readvariableop4savev2_sgd_dense_1_bias_momentum_read_readvariableop6savev2_sgd_dense_2_kernel_momentum_read_readvariableop4savev2_sgd_dense_2_bias_momentum_read_readvariableopAsavev2_sgd_convblock_0_conv2d_kernel_momentum_read_readvariableop?savev2_sgd_convblock_0_conv2d_bias_momentum_read_readvariableopCsavev2_sgd_convblock_0_conv2d_1_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_0_conv2d_1_bias_momentum_read_readvariableopMsavev2_sgd_convblock_0_batch_normalization_gamma_momentum_read_readvariableopLsavev2_sgd_convblock_0_batch_normalization_beta_momentum_read_readvariableopOsavev2_sgd_convblock_0_batch_normalization_1_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_0_batch_normalization_1_beta_momentum_read_readvariableopCsavev2_sgd_convblock_1_conv2d_2_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_1_conv2d_2_bias_momentum_read_readvariableopCsavev2_sgd_convblock_1_conv2d_3_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_1_conv2d_3_bias_momentum_read_readvariableopOsavev2_sgd_convblock_1_batch_normalization_2_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_1_batch_normalization_2_beta_momentum_read_readvariableopOsavev2_sgd_convblock_1_batch_normalization_3_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_1_batch_normalization_3_beta_momentum_read_readvariableopCsavev2_sgd_convblock_2_conv2d_4_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_2_conv2d_4_bias_momentum_read_readvariableopCsavev2_sgd_convblock_2_conv2d_5_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_2_conv2d_5_bias_momentum_read_readvariableopOsavev2_sgd_convblock_2_batch_normalization_4_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_2_batch_normalization_4_beta_momentum_read_readvariableopOsavev2_sgd_convblock_2_batch_normalization_5_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_2_batch_normalization_5_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *e
dtypes[
Y2W		2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
��:�:	�:: : : : : : :  : : : : : : : : : :		 @:@:		@@:@:@:@:@:@:@:@:@:@:@�:�:��:�:�:�:�:�:�:�:�:�:: : : : ::::::
��:�:
��:�:	�:: : :  : : : : : :		 @:@:		@@:@:@:@:@:@:@�:�:��:�:�:�:�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:		 @: 

_output_shapes
:@:,(
&
_output_shapes
:		@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:  

_output_shapes
:@: !

_output_shapes
:@: "

_output_shapes
:@:-#)
'
_output_shapes
:@�:!$

_output_shapes	
:�:.%*
(
_output_shapes
:��:!&

_output_shapes	
:�:!'

_output_shapes	
:�:!(

_output_shapes	
:�:!)

_output_shapes	
:�:!*

_output_shapes	
:�:!+

_output_shapes	
:�:!,

_output_shapes	
:�:!-

_output_shapes	
:�:!.

_output_shapes	
:�: /

_output_shapes
::0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: : 4

_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
::$8 

_output_shapes

::&9"
 
_output_shapes
:
��:!:

_output_shapes	
:�:&;"
 
_output_shapes
:
��:!<

_output_shapes	
:�:%=!

_output_shapes
:	�: >

_output_shapes
::,?(
&
_output_shapes
: : @

_output_shapes
: :,A(
&
_output_shapes
:  : B

_output_shapes
: : C

_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: : F

_output_shapes
: :,G(
&
_output_shapes
:		 @: H

_output_shapes
:@:,I(
&
_output_shapes
:		@@: J

_output_shapes
:@: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@: N

_output_shapes
:@:-O)
'
_output_shapes
:@�:!P

_output_shapes	
:�:.Q*
(
_output_shapes
:��:!R

_output_shapes	
:�:!S

_output_shapes	
:�:!T

_output_shapes	
:�:!U

_output_shapes	
:�:!V

_output_shapes	
:�:W

_output_shapes
: 
�
{
__inference_loss_fn_6_416127J
Fconvblock_1_conv2d_3_kernel_regularizer_square_readvariableop_resource
identity��
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_1_conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulr
IdentityIdentity/ConvBlock-1/conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
�
6__inference_batch_normalization_5_layer_call_fn_416625

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4123852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
p
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_413358

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_416574

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4123122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
,__inference_ConvBlock-0_layer_call_fn_415481
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������ll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_4127152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������ll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:�����������

_user_specified_namex
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_412416

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�Q
�
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_415423
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity��
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv *
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������vv 2
activation/Relu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������vv : : : : :*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll *
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll 2
conv2d_1/BiasAdd�
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ll 2
activation_1/Relu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������ll : : : : :*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������ll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������:::::::::::::T P
1
_output_shapes
:�����������

_user_specified_namex
�d
�
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_412875
x+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity��$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@2
conv2d_2/BiasAdd�
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������..@2
activation_2/Relu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������..@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_2/FusedBatchNormV3�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@2
conv2d_3/BiasAdd�
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������&&@2
activation_3/Relu�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������&&@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_3/FusedBatchNormV3�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:���������&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������66 ::::::::::::2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:R N
/
_output_shapes
:���������66 

_user_specified_namex
��
�
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_414710

inputs=
9random_rotation_stateful_uniform_statefuluniform_resource5
1convblock_0_conv2d_conv2d_readvariableop_resource6
2convblock_0_conv2d_biasadd_readvariableop_resource;
7convblock_0_batch_normalization_readvariableop_resource=
9convblock_0_batch_normalization_readvariableop_1_resourceL
Hconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceN
Jconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource7
3convblock_0_conv2d_1_conv2d_readvariableop_resource8
4convblock_0_conv2d_1_biasadd_readvariableop_resource=
9convblock_0_batch_normalization_1_readvariableop_resource?
;convblock_0_batch_normalization_1_readvariableop_1_resourceN
Jconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3convblock_1_conv2d_2_conv2d_readvariableop_resource8
4convblock_1_conv2d_2_biasadd_readvariableop_resource=
9convblock_1_batch_normalization_2_readvariableop_resource?
;convblock_1_batch_normalization_2_readvariableop_1_resourceN
Jconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3convblock_1_conv2d_3_conv2d_readvariableop_resource8
4convblock_1_conv2d_3_biasadd_readvariableop_resource=
9convblock_1_batch_normalization_3_readvariableop_resource?
;convblock_1_batch_normalization_3_readvariableop_1_resourceN
Jconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3convblock_2_conv2d_4_conv2d_readvariableop_resource8
4convblock_2_conv2d_4_biasadd_readvariableop_resource=
9convblock_2_batch_normalization_4_readvariableop_resource?
;convblock_2_batch_normalization_4_readvariableop_1_resourceN
Jconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3convblock_2_conv2d_5_conv2d_readvariableop_resource8
4convblock_2_conv2d_5_biasadd_readvariableop_resource=
9convblock_2_batch_normalization_5_readvariableop_resource?
;convblock_2_batch_normalization_5_readvariableop_1_resourceN
Jconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��.ConvBlock-0/batch_normalization/AssignNewValue�0ConvBlock-0/batch_normalization/AssignNewValue_1�0ConvBlock-0/batch_normalization_1/AssignNewValue�2ConvBlock-0/batch_normalization_1/AssignNewValue_1�0ConvBlock-1/batch_normalization_2/AssignNewValue�2ConvBlock-1/batch_normalization_2/AssignNewValue_1�0ConvBlock-1/batch_normalization_3/AssignNewValue�2ConvBlock-1/batch_normalization_3/AssignNewValue_1�0ConvBlock-2/batch_normalization_4/AssignNewValue�2ConvBlock-2/batch_normalization_4/AssignNewValue_1�0ConvBlock-2/batch_normalization_5/AssignNewValue�2ConvBlock-2/batch_normalization_5/AssignNewValue_1�0random_rotation/stateful_uniform/StatefulUniformd
random_rotation/ShapeShapeinputs*
T0*
_output_shapes
:2
random_rotation/Shape�
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack�
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1�
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2�
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice�
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_1/stack�
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_1�
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2�
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1�
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast�
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_2/stack�
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_1�
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2�
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2�
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast_1�
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&random_rotation/stateful_uniform/shape�
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2&
$random_rotation/stateful_uniform/min�
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2&
$random_rotation/stateful_uniform/max�
:random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:random_rotation/stateful_uniform/StatefulUniform/algorithm�
0random_rotation/stateful_uniform/StatefulUniformStatefulUniform9random_rotation_stateful_uniform_statefuluniform_resourceCrandom_rotation/stateful_uniform/StatefulUniform/algorithm:output:0/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:���������*
shape_dtype022
0random_rotation/stateful_uniform/StatefulUniform�
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/sub�
$random_rotation/stateful_uniform/mulMul9random_rotation/stateful_uniform/StatefulUniform:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:���������2&
$random_rotation/stateful_uniform/mul�
 random_rotation/stateful_uniformAdd(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������2"
 random_rotation/stateful_uniform�
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%random_rotation/rotation_matrix/sub/y�
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub�
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2%
#random_rotation/rotation_matrix/Cos�
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'random_rotation/rotation_matrix/sub_1/y�
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1�
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:���������2%
#random_rotation/rotation_matrix/mul�
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2%
#random_rotation/rotation_matrix/Sin�
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'random_rotation/rotation_matrix/sub_2/y�
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2�
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/mul_1�
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/sub_3�
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/sub_4�
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/y�
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������2)
'random_rotation/rotation_matrix/truediv�
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'random_rotation/rotation_matrix/sub_5/y�
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5�
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/Sin_1�
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'random_rotation/rotation_matrix/sub_6/y�
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6�
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/mul_2�
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/Cos_1�
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'random_rotation/rotation_matrix/sub_7/y�
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7�
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/mul_3�
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:���������2%
#random_rotation/rotation_matrix/add�
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/sub_8�
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/y�
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:���������2+
)random_rotation/rotation_matrix/truediv_1�
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape�
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack�
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1�
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2�
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice�
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/Cos_2�
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stack�
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1�
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2�
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1�
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/Sin_2�
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stack�
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1�
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2�
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2�
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2%
#random_rotation/rotation_matrix/Neg�
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stack�
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1�
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2�
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3�
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/Sin_3�
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stack�
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1�
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2�
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4�
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:���������2'
%random_rotation/rotation_matrix/Cos_3�
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stack�
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1�
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2�
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5�
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stack�
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1�
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2�
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6�
+random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/zeros/mul/y�
)random_rotation/rotation_matrix/zeros/mulMul6random_rotation/rotation_matrix/strided_slice:output:04random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2+
)random_rotation/rotation_matrix/zeros/mul�
,random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2.
,random_rotation/rotation_matrix/zeros/Less/y�
*random_rotation/rotation_matrix/zeros/LessLess-random_rotation/rotation_matrix/zeros/mul:z:05random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2,
*random_rotation/rotation_matrix/zeros/Less�
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1�
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed�
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Const�
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2'
%random_rotation/rotation_matrix/zeros�
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axis�
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2(
&random_rotation/rotation_matrix/concatx
random_rotation/transform/ShapeShapeinputs*
T0*
_output_shapes
:2!
random_rotation/transform/Shape�
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack�
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1�
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2�
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice�
4random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputs/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV2�
(ConvBlock-0/conv2d/Conv2D/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(ConvBlock-0/conv2d/Conv2D/ReadVariableOp�
ConvBlock-0/conv2d/Conv2DConv2DIrandom_rotation/transform/ImageProjectiveTransformV2:transformed_images:00ConvBlock-0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv *
paddingVALID*
strides
2
ConvBlock-0/conv2d/Conv2D�
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOp�
ConvBlock-0/conv2d/BiasAddBiasAdd"ConvBlock-0/conv2d/Conv2D:output:01ConvBlock-0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv 2
ConvBlock-0/conv2d/BiasAdd�
ConvBlock-0/activation/ReluRelu#ConvBlock-0/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������vv 2
ConvBlock-0/activation/Relu�
.ConvBlock-0/batch_normalization/ReadVariableOpReadVariableOp7convblock_0_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype020
.ConvBlock-0/batch_normalization/ReadVariableOp�
0ConvBlock-0/batch_normalization/ReadVariableOp_1ReadVariableOp9convblock_0_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization/ReadVariableOp_1�
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp�
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
0ConvBlock-0/batch_normalization/FusedBatchNormV3FusedBatchNormV3)ConvBlock-0/activation/Relu:activations:06ConvBlock-0/batch_normalization/ReadVariableOp:value:08ConvBlock-0/batch_normalization/ReadVariableOp_1:value:0GConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0IConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������vv : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<22
0ConvBlock-0/batch_normalization/FusedBatchNormV3�
.ConvBlock-0/batch_normalization/AssignNewValueAssignVariableOpHconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource=ConvBlock-0/batch_normalization/FusedBatchNormV3:batch_mean:0@^ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp*[
_classQ
OMloc:@ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype020
.ConvBlock-0/batch_normalization/AssignNewValue�
0ConvBlock-0/batch_normalization/AssignNewValue_1AssignVariableOpJconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceAConvBlock-0/batch_normalization/FusedBatchNormV3:batch_variance:0B^ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*]
_classS
QOloc:@ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype022
0ConvBlock-0/batch_normalization/AssignNewValue_1�
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp�
ConvBlock-0/conv2d_1/Conv2DConv2D4ConvBlock-0/batch_normalization/FusedBatchNormV3:y:02ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll *
paddingVALID*
strides
2
ConvBlock-0/conv2d_1/Conv2D�
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp�
ConvBlock-0/conv2d_1/BiasAddBiasAdd$ConvBlock-0/conv2d_1/Conv2D:output:03ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll 2
ConvBlock-0/conv2d_1/BiasAdd�
ConvBlock-0/activation_1/ReluRelu%ConvBlock-0/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ll 2
ConvBlock-0/activation_1/Relu�
0ConvBlock-0/batch_normalization_1/ReadVariableOpReadVariableOp9convblock_0_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization_1/ReadVariableOp�
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1ReadVariableOp;convblock_0_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1�
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3+ConvBlock-0/activation_1/Relu:activations:08ConvBlock-0/batch_normalization_1/ReadVariableOp:value:0:ConvBlock-0/batch_normalization_1/ReadVariableOp_1:value:0IConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������ll : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<24
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3�
0ConvBlock-0/batch_normalization_1/AssignNewValueAssignVariableOpJconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource?ConvBlock-0/batch_normalization_1/FusedBatchNormV3:batch_mean:0B^ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-0/batch_normalization_1/AssignNewValue�
2ConvBlock-0/batch_normalization_1/AssignNewValue_1AssignVariableOpLconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-0/batch_normalization_1/FusedBatchNormV3:batch_variance:0D^ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-0/batch_normalization_1/AssignNewValue_1�
max_pooling2d/MaxPoolMaxPool6ConvBlock-0/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:���������66 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool�
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02,
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp�
ConvBlock-1/conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:02ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_2/Conv2D�
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp�
ConvBlock-1/conv2d_2/BiasAddBiasAdd$ConvBlock-1/conv2d_2/Conv2D:output:03ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@2
ConvBlock-1/conv2d_2/BiasAdd�
ConvBlock-1/activation_2/ReluRelu%ConvBlock-1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������..@2
ConvBlock-1/activation_2/Relu�
0ConvBlock-1/batch_normalization_2/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_2/ReadVariableOp�
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1�
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_2/Relu:activations:08ConvBlock-1/batch_normalization_2/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_2/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������..@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<24
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3�
0ConvBlock-1/batch_normalization_2/AssignNewValueAssignVariableOpJconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource?ConvBlock-1/batch_normalization_2/FusedBatchNormV3:batch_mean:0B^ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-1/batch_normalization_2/AssignNewValue�
2ConvBlock-1/batch_normalization_2/AssignNewValue_1AssignVariableOpLconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-1/batch_normalization_2/FusedBatchNormV3:batch_variance:0D^ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-1/batch_normalization_2/AssignNewValue_1�
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02,
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp�
ConvBlock-1/conv2d_3/Conv2DConv2D6ConvBlock-1/batch_normalization_2/FusedBatchNormV3:y:02ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_3/Conv2D�
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp�
ConvBlock-1/conv2d_3/BiasAddBiasAdd$ConvBlock-1/conv2d_3/Conv2D:output:03ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@2
ConvBlock-1/conv2d_3/BiasAdd�
ConvBlock-1/activation_3/ReluRelu%ConvBlock-1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������&&@2
ConvBlock-1/activation_3/Relu�
0ConvBlock-1/batch_normalization_3/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_3/ReadVariableOp�
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1�
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_3/Relu:activations:08ConvBlock-1/batch_normalization_3/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_3/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������&&@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<24
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3�
0ConvBlock-1/batch_normalization_3/AssignNewValueAssignVariableOpJconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?ConvBlock-1/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-1/batch_normalization_3/AssignNewValue�
2ConvBlock-1/batch_normalization_3/AssignNewValue_1AssignVariableOpLconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-1/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-1/batch_normalization_3/AssignNewValue_1�
max_pooling2d_1/MaxPoolMaxPool6ConvBlock-1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool�
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02,
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp�
ConvBlock-2/conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:02ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
ConvBlock-2/conv2d_4/Conv2D�
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp�
ConvBlock-2/conv2d_4/BiasAddBiasAdd$ConvBlock-2/conv2d_4/Conv2D:output:03ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/conv2d_4/BiasAdd�
ConvBlock-2/activation_4/ReluRelu%ConvBlock-2/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/activation_4/Relu�
0ConvBlock-2/batch_normalization_4/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype022
0ConvBlock-2/batch_normalization_4/ReadVariableOp�
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype024
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1�
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02C
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02E
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_4/Relu:activations:08ConvBlock-2/batch_normalization_4/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_4/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<24
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3�
0ConvBlock-2/batch_normalization_4/AssignNewValueAssignVariableOpJconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?ConvBlock-2/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-2/batch_normalization_4/AssignNewValue�
2ConvBlock-2/batch_normalization_4/AssignNewValue_1AssignVariableOpLconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-2/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-2/batch_normalization_4/AssignNewValue_1�
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02,
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp�
ConvBlock-2/conv2d_5/Conv2DConv2D6ConvBlock-2/batch_normalization_4/FusedBatchNormV3:y:02ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
ConvBlock-2/conv2d_5/Conv2D�
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp�
ConvBlock-2/conv2d_5/BiasAddBiasAdd$ConvBlock-2/conv2d_5/Conv2D:output:03ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/conv2d_5/BiasAdd�
ConvBlock-2/activation_5/ReluRelu%ConvBlock-2/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/activation_5/Relu�
0ConvBlock-2/batch_normalization_5/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype022
0ConvBlock-2/batch_normalization_5/ReadVariableOp�
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype024
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1�
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02C
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02E
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_5/Relu:activations:08ConvBlock-2/batch_normalization_5/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_5/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<24
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3�
0ConvBlock-2/batch_normalization_5/AssignNewValueAssignVariableOpJconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?ConvBlock-2/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-2/batch_normalization_5/AssignNewValue�
2ConvBlock-2/batch_normalization_5/AssignNewValue_1AssignVariableOpLconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-2/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-2/batch_normalization_5/AssignNewValue_1�
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices�
global_average_pooling2d/MeanMean6ConvBlock-2/batch_normalization_5/FusedBatchNormV3:y:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling2d/Mean�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd�
!monte_carlo_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2#
!monte_carlo_dropout/dropout/Const�
monte_carlo_dropout/dropout/MulMuldense/BiasAdd:output:0*monte_carlo_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2!
monte_carlo_dropout/dropout/Mul�
!monte_carlo_dropout/dropout/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2#
!monte_carlo_dropout/dropout/Shape�
8monte_carlo_dropout/dropout/random_uniform/RandomUniformRandomUniform*monte_carlo_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"2:
8monte_carlo_dropout/dropout/random_uniform/RandomUniform�
*monte_carlo_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*monte_carlo_dropout/dropout/GreaterEqual/y�
(monte_carlo_dropout/dropout/GreaterEqualGreaterEqualAmonte_carlo_dropout/dropout/random_uniform/RandomUniform:output:03monte_carlo_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2*
(monte_carlo_dropout/dropout/GreaterEqual�
 monte_carlo_dropout/dropout/CastCast,monte_carlo_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2"
 monte_carlo_dropout/dropout/Cast�
!monte_carlo_dropout/dropout/Mul_1Mul#monte_carlo_dropout/dropout/Mul:z:0$monte_carlo_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2#
!monte_carlo_dropout/dropout/Mul_1�
activation_6/ReluRelu%monte_carlo_dropout/dropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2
activation_6/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulactivation_6/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAdd�
#monte_carlo_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#monte_carlo_dropout_1/dropout/Const�
!monte_carlo_dropout_1/dropout/MulMuldense_1/BiasAdd:output:0,monte_carlo_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������2#
!monte_carlo_dropout_1/dropout/Mul�
#monte_carlo_dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2%
#monte_carlo_dropout_1/dropout/Shape�
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniformRandomUniform,monte_carlo_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"*
seed22<
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniform�
,monte_carlo_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,monte_carlo_dropout_1/dropout/GreaterEqual/y�
*monte_carlo_dropout_1/dropout/GreaterEqualGreaterEqualCmonte_carlo_dropout_1/dropout/random_uniform/RandomUniform:output:05monte_carlo_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2,
*monte_carlo_dropout_1/dropout/GreaterEqual�
"monte_carlo_dropout_1/dropout/CastCast.monte_carlo_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2$
"monte_carlo_dropout_1/dropout/Cast�
#monte_carlo_dropout_1/dropout/Mul_1Mul%monte_carlo_dropout_1/dropout/Mul:z:0&monte_carlo_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2%
#monte_carlo_dropout_1/dropout/Mul_1�
activation_7/ReluRelu'monte_carlo_dropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2
activation_7/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulactivation_7/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_2/Sigmoid�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentitydense_2/Sigmoid:y:0/^ConvBlock-0/batch_normalization/AssignNewValue1^ConvBlock-0/batch_normalization/AssignNewValue_11^ConvBlock-0/batch_normalization_1/AssignNewValue3^ConvBlock-0/batch_normalization_1/AssignNewValue_11^ConvBlock-1/batch_normalization_2/AssignNewValue3^ConvBlock-1/batch_normalization_2/AssignNewValue_11^ConvBlock-1/batch_normalization_3/AssignNewValue3^ConvBlock-1/batch_normalization_3/AssignNewValue_11^ConvBlock-2/batch_normalization_4/AssignNewValue3^ConvBlock-2/batch_normalization_4/AssignNewValue_11^ConvBlock-2/batch_normalization_5/AssignNewValue3^ConvBlock-2/batch_normalization_5/AssignNewValue_11^random_rotation/stateful_uniform/StatefulUniform*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:::::::::::::::::::::::::::::::::::::::::::2`
.ConvBlock-0/batch_normalization/AssignNewValue.ConvBlock-0/batch_normalization/AssignNewValue2d
0ConvBlock-0/batch_normalization/AssignNewValue_10ConvBlock-0/batch_normalization/AssignNewValue_12d
0ConvBlock-0/batch_normalization_1/AssignNewValue0ConvBlock-0/batch_normalization_1/AssignNewValue2h
2ConvBlock-0/batch_normalization_1/AssignNewValue_12ConvBlock-0/batch_normalization_1/AssignNewValue_12d
0ConvBlock-1/batch_normalization_2/AssignNewValue0ConvBlock-1/batch_normalization_2/AssignNewValue2h
2ConvBlock-1/batch_normalization_2/AssignNewValue_12ConvBlock-1/batch_normalization_2/AssignNewValue_12d
0ConvBlock-1/batch_normalization_3/AssignNewValue0ConvBlock-1/batch_normalization_3/AssignNewValue2h
2ConvBlock-1/batch_normalization_3/AssignNewValue_12ConvBlock-1/batch_normalization_3/AssignNewValue_12d
0ConvBlock-2/batch_normalization_4/AssignNewValue0ConvBlock-2/batch_normalization_4/AssignNewValue2h
2ConvBlock-2/batch_normalization_4/AssignNewValue_12ConvBlock-2/batch_normalization_4/AssignNewValue_12d
0ConvBlock-2/batch_normalization_5/AssignNewValue0ConvBlock-2/batch_normalization_5/AssignNewValue2h
2ConvBlock-2/batch_normalization_5/AssignNewValue_12ConvBlock-2/batch_normalization_5/AssignNewValue_12d
0random_rotation/stateful_uniform/StatefulUniform0random_rotation/stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_3_layer_call_fn_416473

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4121652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�c
�
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_412645
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv *
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������vv 2
activation/Relu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������vv : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2&
$batch_normalization/FusedBatchNormV3�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll *
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll 2
conv2d_1/BiasAdd�
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ll 2
activation_1/Relu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������ll : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_1/FusedBatchNormV3�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1*
T0*/
_output_shapes
:���������ll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_1:T P
1
_output_shapes
:�����������

_user_specified_namex
�
�
4__inference_batch_normalization_layer_call_fn_416270

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4118722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
y
__inference_loss_fn_0_416061H
Dconvblock_0_conv2d_kernel_regularizer_square_readvariableop_resource
identity��
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_0_conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulp
IdentityIdentity-ConvBlock-0/conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_416442

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_5_layer_call_fn_416638

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4124162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_413390

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
,__inference_ConvBlock-1_layer_call_fn_415707
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_4129452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������66 ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������66 

_user_specified_namex
�
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_412434

inputs
identity�
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_414353
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_4117792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
��
�

N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_414097

inputs
convblock_0_413927
convblock_0_413929
convblock_0_413931
convblock_0_413933
convblock_0_413935
convblock_0_413937
convblock_0_413939
convblock_0_413941
convblock_0_413943
convblock_0_413945
convblock_0_413947
convblock_0_413949
convblock_1_413953
convblock_1_413955
convblock_1_413957
convblock_1_413959
convblock_1_413961
convblock_1_413963
convblock_1_413965
convblock_1_413967
convblock_1_413969
convblock_1_413971
convblock_1_413973
convblock_1_413975
convblock_2_413979
convblock_2_413981
convblock_2_413983
convblock_2_413985
convblock_2_413987
convblock_2_413989
convblock_2_413991
convblock_2_413993
convblock_2_413995
convblock_2_413997
convblock_2_413999
convblock_2_414001
dense_414005
dense_414007
dense_1_414012
dense_1_414014
dense_2_414019
dense_2_414021
identity��#ConvBlock-0/StatefulPartitionedCall�#ConvBlock-1/StatefulPartitionedCall�#ConvBlock-2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+monte_carlo_dropout/StatefulPartitionedCall�-monte_carlo_dropout_1/StatefulPartitionedCall�
random_rotation/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_4125542!
random_rotation/PartitionedCall�
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall(random_rotation/PartitionedCall:output:0convblock_0_413927convblock_0_413929convblock_0_413931convblock_0_413933convblock_0_413935convblock_0_413937convblock_0_413939convblock_0_413941convblock_0_413943convblock_0_413945convblock_0_413947convblock_0_413949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������ll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_4127152%
#ConvBlock-0/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4119932
max_pooling2d/PartitionedCall�
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_413953convblock_1_413955convblock_1_413957convblock_1_413959convblock_1_413961convblock_1_413963convblock_1_413965convblock_1_413967convblock_1_413969convblock_1_413971convblock_1_413973convblock_1_413975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_4129452%
#ConvBlock-1/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4122132!
max_pooling2d_1/PartitionedCall�
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_413979convblock_2_413981convblock_2_413983convblock_2_413985convblock_2_413987convblock_2_413989convblock_2_413991convblock_2_413993convblock_2_413995convblock_2_413997convblock_2_413999convblock_2_414001*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_4131752%
#ConvBlock-2/StatefulPartitionedCall�
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4124342*
(global_average_pooling2d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_414005dense_414007*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4132712
dense/StatefulPartitionedCall�
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_4132992-
+monte_carlo_dropout/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_4133122
activation_6/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_414012dense_1_414014*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4133302!
dense_1/StatefulPartitionedCall�
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_4133582/
-monte_carlo_dropout_1/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_4133712
activation_7/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_414019dense_2_414021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4133902!
dense_2/StatefulPartitionedCall�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413927*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413929*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413939*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413941*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413953*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413955*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413965*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413967*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413979*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413981*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413991*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413993*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::2J
#ConvBlock-0/StatefulPartitionedCall#ConvBlock-0/StatefulPartitionedCall2J
#ConvBlock-1/StatefulPartitionedCall#ConvBlock-1/StatefulPartitionedCall2J
#ConvBlock-2/StatefulPartitionedCall#ConvBlock-2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+monte_carlo_dropout/StatefulPartitionedCall+monte_carlo_dropout/StatefulPartitionedCall2^
-monte_carlo_dropout_1/StatefulPartitionedCall-monte_carlo_dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_416548

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
v
0__inference_random_rotation_layer_call_fn_415250

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_4125502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_2_layer_call_fn_416422

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4120922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�c
�
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_415353
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv *
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������vv 2
activation/Relu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������vv : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2&
$batch_normalization/FusedBatchNormV3�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll *
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll 2
conv2d_1/BiasAdd�
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ll 2
activation_1/Relu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������ll : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_1/FusedBatchNormV3�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1*
T0*/
_output_shapes
:���������ll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_1:T P
1
_output_shapes
:�����������

_user_specified_namex
�d
�
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_413105
x+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity��$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_4/BiasAdd�
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_4/Relu�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_4/FusedBatchNormV3�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_5/Conv2D�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_5/BiasAdd�
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_5/Relu�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_5/ReadVariableOp_1�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_5/FusedBatchNormV3�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������@::::::::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_1:R N
/
_output_shapes
:���������@

_user_specified_namex
�
�
3__inference_FERREIRA2020_class_layer_call_fn_415044

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
 !"#&'()*+*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_4138322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_412385

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
{
&__inference_dense_layer_call_fn_415952

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4132712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_412281

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
�
K__inference_random_rotation_layer_call_and_return_conditional_losses_412550

inputs-
)stateful_uniform_statefuluniform_resource
identity�� stateful_uniform/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2
stateful_uniform/max�
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm�
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:���������*
shape_dtype02"
 stateful_uniform/StatefulUniform�
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub�
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:���������2
stateful_uniform/mul�
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_1/y�
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1�
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_2/y�
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2�
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mul_1�
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/sub_3�
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/y�
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_5/y�
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_6/y�
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6�
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_7/y�
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7�
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mul_3�
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/add�
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y�
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:���������2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape�
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack�
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1�
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2�
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cos_2�
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack�
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1�
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2�
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sin_2�
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack�
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1�
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2�
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2�
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2
rotation_matrix/Neg�
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack�
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1�
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2�
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sin_3�
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack�
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1�
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2�
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cos_3�
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack�
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1�
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2�
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5�
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack�
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1�
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2�
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6|
rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/zeros/mul/y�
rotation_matrix/zeros/mulMul&rotation_matrix/strided_slice:output:0$rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/mul
rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
rotation_matrix/zeros/Less/y�
rotation_matrix/zeros/LessLessrotation_matrix/zeros/mul:z:0%rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/Less�
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1�
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Const�
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis�
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape�
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack�
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1�
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice�
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2�
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�

N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_413653
input_1
convblock_0_413483
convblock_0_413485
convblock_0_413487
convblock_0_413489
convblock_0_413491
convblock_0_413493
convblock_0_413495
convblock_0_413497
convblock_0_413499
convblock_0_413501
convblock_0_413503
convblock_0_413505
convblock_1_413509
convblock_1_413511
convblock_1_413513
convblock_1_413515
convblock_1_413517
convblock_1_413519
convblock_1_413521
convblock_1_413523
convblock_1_413525
convblock_1_413527
convblock_1_413529
convblock_1_413531
convblock_2_413535
convblock_2_413537
convblock_2_413539
convblock_2_413541
convblock_2_413543
convblock_2_413545
convblock_2_413547
convblock_2_413549
convblock_2_413551
convblock_2_413553
convblock_2_413555
convblock_2_413557
dense_413561
dense_413563
dense_1_413568
dense_1_413570
dense_2_413575
dense_2_413577
identity��#ConvBlock-0/StatefulPartitionedCall�#ConvBlock-1/StatefulPartitionedCall�#ConvBlock-2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+monte_carlo_dropout/StatefulPartitionedCall�-monte_carlo_dropout_1/StatefulPartitionedCall�
random_rotation/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_4125542!
random_rotation/PartitionedCall�
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall(random_rotation/PartitionedCall:output:0convblock_0_413483convblock_0_413485convblock_0_413487convblock_0_413489convblock_0_413491convblock_0_413493convblock_0_413495convblock_0_413497convblock_0_413499convblock_0_413501convblock_0_413503convblock_0_413505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������ll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_4127152%
#ConvBlock-0/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4119932
max_pooling2d/PartitionedCall�
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_413509convblock_1_413511convblock_1_413513convblock_1_413515convblock_1_413517convblock_1_413519convblock_1_413521convblock_1_413523convblock_1_413525convblock_1_413527convblock_1_413529convblock_1_413531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_4129452%
#ConvBlock-1/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4122132!
max_pooling2d_1/PartitionedCall�
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_413535convblock_2_413537convblock_2_413539convblock_2_413541convblock_2_413543convblock_2_413545convblock_2_413547convblock_2_413549convblock_2_413551convblock_2_413553convblock_2_413555convblock_2_413557*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_4131752%
#ConvBlock-2/StatefulPartitionedCall�
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4124342*
(global_average_pooling2d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_413561dense_413563*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4132712
dense/StatefulPartitionedCall�
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_4132992-
+monte_carlo_dropout/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_4133122
activation_6/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_413568dense_1_413570*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4133302!
dense_1/StatefulPartitionedCall�
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_4133582/
-monte_carlo_dropout_1/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_4133712
activation_7/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_413575dense_2_413577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4133902!
dense_2/StatefulPartitionedCall�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413483*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413485*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413495*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413497*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413509*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413511*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413521*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413523*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413535*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413537*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413547*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413549*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::2J
#ConvBlock-0/StatefulPartitionedCall#ConvBlock-0/StatefulPartitionedCall2J
#ConvBlock-1/StatefulPartitionedCall#ConvBlock-1/StatefulPartitionedCall2J
#ConvBlock-2/StatefulPartitionedCall#ConvBlock-2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+monte_carlo_dropout/StatefulPartitionedCall+monte_carlo_dropout/StatefulPartitionedCall2^
-monte_carlo_dropout_1/StatefulPartitionedCall-monte_carlo_dropout_1/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
��
�
!__inference__wrapped_model_411779
input_1H
Dferreira2020_class_convblock_0_conv2d_conv2d_readvariableop_resourceI
Eferreira2020_class_convblock_0_conv2d_biasadd_readvariableop_resourceN
Jferreira2020_class_convblock_0_batch_normalization_readvariableop_resourceP
Lferreira2020_class_convblock_0_batch_normalization_readvariableop_1_resource_
[ferreira2020_class_convblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resourcea
]ferreira2020_class_convblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceJ
Fferreira2020_class_convblock_0_conv2d_1_conv2d_readvariableop_resourceK
Gferreira2020_class_convblock_0_conv2d_1_biasadd_readvariableop_resourceP
Lferreira2020_class_convblock_0_batch_normalization_1_readvariableop_resourceR
Nferreira2020_class_convblock_0_batch_normalization_1_readvariableop_1_resourcea
]ferreira2020_class_convblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resourcec
_ferreira2020_class_convblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceJ
Fferreira2020_class_convblock_1_conv2d_2_conv2d_readvariableop_resourceK
Gferreira2020_class_convblock_1_conv2d_2_biasadd_readvariableop_resourceP
Lferreira2020_class_convblock_1_batch_normalization_2_readvariableop_resourceR
Nferreira2020_class_convblock_1_batch_normalization_2_readvariableop_1_resourcea
]ferreira2020_class_convblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourcec
_ferreira2020_class_convblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceJ
Fferreira2020_class_convblock_1_conv2d_3_conv2d_readvariableop_resourceK
Gferreira2020_class_convblock_1_conv2d_3_biasadd_readvariableop_resourceP
Lferreira2020_class_convblock_1_batch_normalization_3_readvariableop_resourceR
Nferreira2020_class_convblock_1_batch_normalization_3_readvariableop_1_resourcea
]ferreira2020_class_convblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourcec
_ferreira2020_class_convblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceJ
Fferreira2020_class_convblock_2_conv2d_4_conv2d_readvariableop_resourceK
Gferreira2020_class_convblock_2_conv2d_4_biasadd_readvariableop_resourceP
Lferreira2020_class_convblock_2_batch_normalization_4_readvariableop_resourceR
Nferreira2020_class_convblock_2_batch_normalization_4_readvariableop_1_resourcea
]ferreira2020_class_convblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resourcec
_ferreira2020_class_convblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceJ
Fferreira2020_class_convblock_2_conv2d_5_conv2d_readvariableop_resourceK
Gferreira2020_class_convblock_2_conv2d_5_biasadd_readvariableop_resourceP
Lferreira2020_class_convblock_2_batch_normalization_5_readvariableop_resourceR
Nferreira2020_class_convblock_2_batch_normalization_5_readvariableop_1_resourcea
]ferreira2020_class_convblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resourcec
_ferreira2020_class_convblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource;
7ferreira2020_class_dense_matmul_readvariableop_resource<
8ferreira2020_class_dense_biasadd_readvariableop_resource=
9ferreira2020_class_dense_1_matmul_readvariableop_resource>
:ferreira2020_class_dense_1_biasadd_readvariableop_resource=
9ferreira2020_class_dense_2_matmul_readvariableop_resource>
:ferreira2020_class_dense_2_biasadd_readvariableop_resource
identity��
;FERREIRA2020_class/ConvBlock-0/conv2d/Conv2D/ReadVariableOpReadVariableOpDferreira2020_class_convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;FERREIRA2020_class/ConvBlock-0/conv2d/Conv2D/ReadVariableOp�
,FERREIRA2020_class/ConvBlock-0/conv2d/Conv2DConv2Dinput_1CFERREIRA2020_class/ConvBlock-0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv *
paddingVALID*
strides
2.
,FERREIRA2020_class/ConvBlock-0/conv2d/Conv2D�
<FERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd/ReadVariableOpReadVariableOpEferreira2020_class_convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<FERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd/ReadVariableOp�
-FERREIRA2020_class/ConvBlock-0/conv2d/BiasAddBiasAdd5FERREIRA2020_class/ConvBlock-0/conv2d/Conv2D:output:0DFERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv 2/
-FERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd�
.FERREIRA2020_class/ConvBlock-0/activation/ReluRelu6FERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������vv 20
.FERREIRA2020_class/ConvBlock-0/activation/Relu�
AFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOpReadVariableOpJferreira2020_class_convblock_0_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02C
AFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp�
CFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp_1ReadVariableOpLferreira2020_class_convblock_0_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02E
CFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp_1�
RFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp[ferreira2020_class_convblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02T
RFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp�
TFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]ferreira2020_class_convblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02V
TFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
CFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3FusedBatchNormV3<FERREIRA2020_class/ConvBlock-0/activation/Relu:activations:0IFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp:value:0KFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp_1:value:0ZFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0\FERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������vv : : : : :*
epsilon%o�:*
is_training( 2E
CFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3�
=FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp�
.FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2DConv2DGFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3:y:0EFERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll *
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D�
>FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp�
/FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D:output:0FFERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll 21
/FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd�
0FERREIRA2020_class/ConvBlock-0/activation_1/ReluRelu8FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ll 22
0FERREIRA2020_class/ConvBlock-0/activation_1/Relu�
CFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOpReadVariableOpLferreira2020_class_convblock_0_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02E
CFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp�
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_0_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02G
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp_1�
TFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02V
TFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
VFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02X
VFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-0/activation_1/Relu:activations:0KFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������ll : : : : :*
epsilon%o�:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3�
(FERREIRA2020_class/max_pooling2d/MaxPoolMaxPoolIFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:���������66 *
ksize
*
paddingVALID*
strides
2*
(FERREIRA2020_class/max_pooling2d/MaxPool�
=FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp�
.FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2DConv2D1FERREIRA2020_class/max_pooling2d/MaxPool:output:0EFERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D�
>FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02@
>FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp�
/FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D:output:0FFERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@21
/FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd�
0FERREIRA2020_class/ConvBlock-1/activation_2/ReluRelu8FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������..@22
0FERREIRA2020_class/ConvBlock-1/activation_2/Relu�
CFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOpReadVariableOpLferreira2020_class_convblock_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02E
CFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp�
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp_1�
TFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02V
TFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
VFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02X
VFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-1/activation_2/Relu:activations:0KFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������..@:@:@:@:@:*
epsilon%o�:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3�
=FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp�
.FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2DConv2DIFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3:y:0EFERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D�
>FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02@
>FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp�
/FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D:output:0FFERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@21
/FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd�
0FERREIRA2020_class/ConvBlock-1/activation_3/ReluRelu8FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������&&@22
0FERREIRA2020_class/ConvBlock-1/activation_3/Relu�
CFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOpReadVariableOpLferreira2020_class_convblock_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02E
CFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp�
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp_1�
TFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02V
TFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
VFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02X
VFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-1/activation_3/Relu:activations:0KFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������&&@:@:@:@:@:*
epsilon%o�:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3�
*FERREIRA2020_class/max_pooling2d_1/MaxPoolMaxPoolIFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2,
*FERREIRA2020_class/max_pooling2d_1/MaxPool�
=FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp�
.FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2DConv2D3FERREIRA2020_class/max_pooling2d_1/MaxPool:output:0EFERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D�
>FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp�
/FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D:output:0FFERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������21
/FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd�
0FERREIRA2020_class/ConvBlock-2/activation_4/ReluRelu8FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������22
0FERREIRA2020_class/ConvBlock-2/activation_4/Relu�
CFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOpReadVariableOpLferreira2020_class_convblock_2_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype02E
CFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp�
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp_1�
TFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02V
TFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
VFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02X
VFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-2/activation_4/Relu:activations:0KFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3�
=FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp�
.FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2DConv2DIFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3:y:0EFERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D�
>FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02@
>FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp�
/FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D:output:0FFERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������21
/FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd�
0FERREIRA2020_class/ConvBlock-2/activation_5/ReluRelu8FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:����������22
0FERREIRA2020_class/ConvBlock-2/activation_5/Relu�
CFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOpReadVariableOpLferreira2020_class_convblock_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype02E
CFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp�
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype02G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp_1�
TFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02V
TFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
VFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02X
VFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-2/activation_5/Relu:activations:0KFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3�
BFERREIRA2020_class/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2D
BFERREIRA2020_class/global_average_pooling2d/Mean/reduction_indices�
0FERREIRA2020_class/global_average_pooling2d/MeanMeanIFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3:y:0KFERREIRA2020_class/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������22
0FERREIRA2020_class/global_average_pooling2d/Mean�
.FERREIRA2020_class/dense/MatMul/ReadVariableOpReadVariableOp7ferreira2020_class_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype020
.FERREIRA2020_class/dense/MatMul/ReadVariableOp�
FERREIRA2020_class/dense/MatMulMatMul9FERREIRA2020_class/global_average_pooling2d/Mean:output:06FERREIRA2020_class/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2!
FERREIRA2020_class/dense/MatMul�
/FERREIRA2020_class/dense/BiasAdd/ReadVariableOpReadVariableOp8ferreira2020_class_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype021
/FERREIRA2020_class/dense/BiasAdd/ReadVariableOp�
 FERREIRA2020_class/dense/BiasAddBiasAdd)FERREIRA2020_class/dense/MatMul:product:07FERREIRA2020_class/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 FERREIRA2020_class/dense/BiasAdd�
4FERREIRA2020_class/monte_carlo_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @26
4FERREIRA2020_class/monte_carlo_dropout/dropout/Const�
2FERREIRA2020_class/monte_carlo_dropout/dropout/MulMul)FERREIRA2020_class/dense/BiasAdd:output:0=FERREIRA2020_class/monte_carlo_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������24
2FERREIRA2020_class/monte_carlo_dropout/dropout/Mul�
4FERREIRA2020_class/monte_carlo_dropout/dropout/ShapeShape)FERREIRA2020_class/dense/BiasAdd:output:0*
T0*
_output_shapes
:26
4FERREIRA2020_class/monte_carlo_dropout/dropout/Shape�
KFERREIRA2020_class/monte_carlo_dropout/dropout/random_uniform/RandomUniformRandomUniform=FERREIRA2020_class/monte_carlo_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"2M
KFERREIRA2020_class/monte_carlo_dropout/dropout/random_uniform/RandomUniform�
=FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2?
=FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual/y�
;FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqualGreaterEqualTFERREIRA2020_class/monte_carlo_dropout/dropout/random_uniform/RandomUniform:output:0FFERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2=
;FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual�
3FERREIRA2020_class/monte_carlo_dropout/dropout/CastCast?FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������25
3FERREIRA2020_class/monte_carlo_dropout/dropout/Cast�
4FERREIRA2020_class/monte_carlo_dropout/dropout/Mul_1Mul6FERREIRA2020_class/monte_carlo_dropout/dropout/Mul:z:07FERREIRA2020_class/monte_carlo_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������26
4FERREIRA2020_class/monte_carlo_dropout/dropout/Mul_1�
$FERREIRA2020_class/activation_6/ReluRelu8FERREIRA2020_class/monte_carlo_dropout/dropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2&
$FERREIRA2020_class/activation_6/Relu�
0FERREIRA2020_class/dense_1/MatMul/ReadVariableOpReadVariableOp9ferreira2020_class_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype022
0FERREIRA2020_class/dense_1/MatMul/ReadVariableOp�
!FERREIRA2020_class/dense_1/MatMulMatMul2FERREIRA2020_class/activation_6/Relu:activations:08FERREIRA2020_class/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!FERREIRA2020_class/dense_1/MatMul�
1FERREIRA2020_class/dense_1/BiasAdd/ReadVariableOpReadVariableOp:ferreira2020_class_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype023
1FERREIRA2020_class/dense_1/BiasAdd/ReadVariableOp�
"FERREIRA2020_class/dense_1/BiasAddBiasAdd+FERREIRA2020_class/dense_1/MatMul:product:09FERREIRA2020_class/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"FERREIRA2020_class/dense_1/BiasAdd�
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @28
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/Const�
4FERREIRA2020_class/monte_carlo_dropout_1/dropout/MulMul+FERREIRA2020_class/dense_1/BiasAdd:output:0?FERREIRA2020_class/monte_carlo_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������26
4FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul�
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/ShapeShape+FERREIRA2020_class/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:28
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/Shape�
MFERREIRA2020_class/monte_carlo_dropout_1/dropout/random_uniform/RandomUniformRandomUniform?FERREIRA2020_class/monte_carlo_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"*
seed22O
MFERREIRA2020_class/monte_carlo_dropout_1/dropout/random_uniform/RandomUniform�
?FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2A
?FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual/y�
=FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqualGreaterEqualVFERREIRA2020_class/monte_carlo_dropout_1/dropout/random_uniform/RandomUniform:output:0HFERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2?
=FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual�
5FERREIRA2020_class/monte_carlo_dropout_1/dropout/CastCastAFERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������27
5FERREIRA2020_class/monte_carlo_dropout_1/dropout/Cast�
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul_1Mul8FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul:z:09FERREIRA2020_class/monte_carlo_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:����������28
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul_1�
$FERREIRA2020_class/activation_7/ReluRelu:FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2&
$FERREIRA2020_class/activation_7/Relu�
0FERREIRA2020_class/dense_2/MatMul/ReadVariableOpReadVariableOp9ferreira2020_class_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype022
0FERREIRA2020_class/dense_2/MatMul/ReadVariableOp�
!FERREIRA2020_class/dense_2/MatMulMatMul2FERREIRA2020_class/activation_7/Relu:activations:08FERREIRA2020_class/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!FERREIRA2020_class/dense_2/MatMul�
1FERREIRA2020_class/dense_2/BiasAdd/ReadVariableOpReadVariableOp:ferreira2020_class_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1FERREIRA2020_class/dense_2/BiasAdd/ReadVariableOp�
"FERREIRA2020_class/dense_2/BiasAddBiasAdd+FERREIRA2020_class/dense_2/MatMul:product:09FERREIRA2020_class/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"FERREIRA2020_class/dense_2/BiasAdd�
"FERREIRA2020_class/dense_2/SigmoidSigmoid+FERREIRA2020_class/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2$
"FERREIRA2020_class/dense_2/Sigmoidz
IdentityIdentity&FERREIRA2020_class/dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:::::::::::::::::::::::::::::::::::::::::::Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_411945

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_416308

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�d
�
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_415805
x+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity��$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_4/BiasAdd�
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_4/Relu�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_4/FusedBatchNormV3�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_5/Conv2D�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_5/BiasAdd�
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_5/Relu�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_5/ReadVariableOp_1�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_5/FusedBatchNormV3�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������@::::::::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_1:R N
/
_output_shapes
:���������@

_user_specified_namex
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_416290

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_416561

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4122812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_6_layer_call_and_return_conditional_losses_415974

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_416378

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_416334

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4119762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_1_layer_call_fn_416321

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4119452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
3__inference_FERREIRA2020_class_layer_call_fn_413921
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
 !"#&'()*+*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_4138322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
d
H__inference_activation_7_layer_call_and_return_conditional_losses_413371

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
K__inference_random_rotation_layer_call_and_return_conditional_losses_415239

inputs-
)stateful_uniform_statefuluniform_resource
identity�� stateful_uniform/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2
stateful_uniform/max�
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm�
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:���������*
shape_dtype02"
 stateful_uniform/StatefulUniform�
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub�
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:���������2
stateful_uniform/mul�
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_1/y�
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1�
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_2/y�
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2�
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mul_1�
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/sub_3�
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/y�
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_5/y�
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_6/y�
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6�
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
rotation_matrix/sub_7/y�
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7�
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/mul_3�
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/add�
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y�
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:���������2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape�
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack�
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1�
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2�
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cos_2�
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack�
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1�
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2�
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sin_2�
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack�
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1�
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2�
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2�
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������2
rotation_matrix/Neg�
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack�
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1�
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2�
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Sin_3�
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack�
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1�
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2�
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������2
rotation_matrix/Cos_3�
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack�
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1�
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2�
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5�
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack�
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1�
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2�
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6|
rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/zeros/mul/y�
rotation_matrix/zeros/mulMul&rotation_matrix/strided_slice:output:0$rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/mul
rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
rotation_matrix/zeros/Less/y�
rotation_matrix/zeros/LessLessrotation_matrix/zeros/mul:z:0%rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/Less�
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1�
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Const�
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis�
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape�
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack�
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1�
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice�
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:�����������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2�
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_411976

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
g
K__inference_random_rotation_layer_call_and_return_conditional_losses_415243

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_412196

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_416612

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�R
�
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_415649
x+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity��
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@2
conv2d_2/BiasAdd�
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������..@2
activation_2/Relu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������..@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@2
conv2d_3/BiasAdd�
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������&&@2
activation_3/Relu�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������&&@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������66 :::::::::::::R N
/
_output_shapes
:���������66 

_user_specified_namex
�
p
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_416010

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
y
__inference_loss_fn_3_416094H
Dconvblock_0_conv2d_1_bias_regularizer_square_readvariableop_resource
identity��
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_0_conv2d_1_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mulp
IdentityIdentity-ConvBlock-0/conv2d_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
{
__inference_loss_fn_8_416149J
Fconvblock_2_conv2d_4_kernel_regularizer_square_readvariableop_resource
identity��
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_2_conv2d_4_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulr
IdentityIdentity/ConvBlock-2/conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
n
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_415964

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_413832

inputs
random_rotation_413659
convblock_0_413662
convblock_0_413664
convblock_0_413666
convblock_0_413668
convblock_0_413670
convblock_0_413672
convblock_0_413674
convblock_0_413676
convblock_0_413678
convblock_0_413680
convblock_0_413682
convblock_0_413684
convblock_1_413688
convblock_1_413690
convblock_1_413692
convblock_1_413694
convblock_1_413696
convblock_1_413698
convblock_1_413700
convblock_1_413702
convblock_1_413704
convblock_1_413706
convblock_1_413708
convblock_1_413710
convblock_2_413714
convblock_2_413716
convblock_2_413718
convblock_2_413720
convblock_2_413722
convblock_2_413724
convblock_2_413726
convblock_2_413728
convblock_2_413730
convblock_2_413732
convblock_2_413734
convblock_2_413736
dense_413740
dense_413742
dense_1_413747
dense_1_413749
dense_2_413754
dense_2_413756
identity��#ConvBlock-0/StatefulPartitionedCall�#ConvBlock-1/StatefulPartitionedCall�#ConvBlock-2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+monte_carlo_dropout/StatefulPartitionedCall�-monte_carlo_dropout_1/StatefulPartitionedCall�'random_rotation/StatefulPartitionedCall�
'random_rotation/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_rotation_413659*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_4125502)
'random_rotation/StatefulPartitionedCall�
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0convblock_0_413662convblock_0_413664convblock_0_413666convblock_0_413668convblock_0_413670convblock_0_413672convblock_0_413674convblock_0_413676convblock_0_413678convblock_0_413680convblock_0_413682convblock_0_413684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������ll **
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_4126452%
#ConvBlock-0/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4119932
max_pooling2d/PartitionedCall�
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_413688convblock_1_413690convblock_1_413692convblock_1_413694convblock_1_413696convblock_1_413698convblock_1_413700convblock_1_413702convblock_1_413704convblock_1_413706convblock_1_413708convblock_1_413710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������&&@**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_4128752%
#ConvBlock-1/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4122132!
max_pooling2d_1/PartitionedCall�
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_413714convblock_2_413716convblock_2_413718convblock_2_413720convblock_2_413722convblock_2_413724convblock_2_413726convblock_2_413728convblock_2_413730convblock_2_413732convblock_2_413734convblock_2_413736*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_4131052%
#ConvBlock-2/StatefulPartitionedCall�
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4124342*
(global_average_pooling2d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_413740dense_413742*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4132712
dense/StatefulPartitionedCall�
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_4132992-
+monte_carlo_dropout/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_4133122
activation_6/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_413747dense_1_413749*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4133302!
dense_1/StatefulPartitionedCall�
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_4133582/
-monte_carlo_dropout_1/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_4133712
activation_7/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_413754dense_2_413756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4133902!
dense_2/StatefulPartitionedCall�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413662*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413664*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413674*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_413676*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413688*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413690*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413700*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413702*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413714*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413716*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413726*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413728*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:::::::::::::::::::::::::::::::::::::::::::2J
#ConvBlock-0/StatefulPartitionedCall#ConvBlock-0/StatefulPartitionedCall2J
#ConvBlock-1/StatefulPartitionedCall#ConvBlock-1/StatefulPartitionedCall2J
#ConvBlock-2/StatefulPartitionedCall#ConvBlock-2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+monte_carlo_dropout/StatefulPartitionedCall+monte_carlo_dropout/StatefulPartitionedCall2^
-monte_carlo_dropout_1/StatefulPartitionedCall-monte_carlo_dropout_1/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_414953

inputs5
1convblock_0_conv2d_conv2d_readvariableop_resource6
2convblock_0_conv2d_biasadd_readvariableop_resource;
7convblock_0_batch_normalization_readvariableop_resource=
9convblock_0_batch_normalization_readvariableop_1_resourceL
Hconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resourceN
Jconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource7
3convblock_0_conv2d_1_conv2d_readvariableop_resource8
4convblock_0_conv2d_1_biasadd_readvariableop_resource=
9convblock_0_batch_normalization_1_readvariableop_resource?
;convblock_0_batch_normalization_1_readvariableop_1_resourceN
Jconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7
3convblock_1_conv2d_2_conv2d_readvariableop_resource8
4convblock_1_conv2d_2_biasadd_readvariableop_resource=
9convblock_1_batch_normalization_2_readvariableop_resource?
;convblock_1_batch_normalization_2_readvariableop_1_resourceN
Jconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7
3convblock_1_conv2d_3_conv2d_readvariableop_resource8
4convblock_1_conv2d_3_biasadd_readvariableop_resource=
9convblock_1_batch_normalization_3_readvariableop_resource?
;convblock_1_batch_normalization_3_readvariableop_1_resourceN
Jconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7
3convblock_2_conv2d_4_conv2d_readvariableop_resource8
4convblock_2_conv2d_4_biasadd_readvariableop_resource=
9convblock_2_batch_normalization_4_readvariableop_resource?
;convblock_2_batch_normalization_4_readvariableop_1_resourceN
Jconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3convblock_2_conv2d_5_conv2d_readvariableop_resource8
4convblock_2_conv2d_5_biasadd_readvariableop_resource=
9convblock_2_batch_normalization_5_readvariableop_resource?
;convblock_2_batch_normalization_5_readvariableop_1_resourceN
Jconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceP
Lconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��
(ConvBlock-0/conv2d/Conv2D/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(ConvBlock-0/conv2d/Conv2D/ReadVariableOp�
ConvBlock-0/conv2d/Conv2DConv2Dinputs0ConvBlock-0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv *
paddingVALID*
strides
2
ConvBlock-0/conv2d/Conv2D�
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOp�
ConvBlock-0/conv2d/BiasAddBiasAdd"ConvBlock-0/conv2d/Conv2D:output:01ConvBlock-0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv 2
ConvBlock-0/conv2d/BiasAdd�
ConvBlock-0/activation/ReluRelu#ConvBlock-0/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������vv 2
ConvBlock-0/activation/Relu�
.ConvBlock-0/batch_normalization/ReadVariableOpReadVariableOp7convblock_0_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype020
.ConvBlock-0/batch_normalization/ReadVariableOp�
0ConvBlock-0/batch_normalization/ReadVariableOp_1ReadVariableOp9convblock_0_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization/ReadVariableOp_1�
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp�
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
0ConvBlock-0/batch_normalization/FusedBatchNormV3FusedBatchNormV3)ConvBlock-0/activation/Relu:activations:06ConvBlock-0/batch_normalization/ReadVariableOp:value:08ConvBlock-0/batch_normalization/ReadVariableOp_1:value:0GConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0IConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������vv : : : : :*
epsilon%o�:*
is_training( 22
0ConvBlock-0/batch_normalization/FusedBatchNormV3�
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp�
ConvBlock-0/conv2d_1/Conv2DConv2D4ConvBlock-0/batch_normalization/FusedBatchNormV3:y:02ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll *
paddingVALID*
strides
2
ConvBlock-0/conv2d_1/Conv2D�
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp�
ConvBlock-0/conv2d_1/BiasAddBiasAdd$ConvBlock-0/conv2d_1/Conv2D:output:03ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll 2
ConvBlock-0/conv2d_1/BiasAdd�
ConvBlock-0/activation_1/ReluRelu%ConvBlock-0/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ll 2
ConvBlock-0/activation_1/Relu�
0ConvBlock-0/batch_normalization_1/ReadVariableOpReadVariableOp9convblock_0_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization_1/ReadVariableOp�
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1ReadVariableOp;convblock_0_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1�
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3+ConvBlock-0/activation_1/Relu:activations:08ConvBlock-0/batch_normalization_1/ReadVariableOp:value:0:ConvBlock-0/batch_normalization_1/ReadVariableOp_1:value:0IConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������ll : : : : :*
epsilon%o�:*
is_training( 24
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3�
max_pooling2d/MaxPoolMaxPool6ConvBlock-0/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:���������66 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool�
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02,
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp�
ConvBlock-1/conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:02ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_2/Conv2D�
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp�
ConvBlock-1/conv2d_2/BiasAddBiasAdd$ConvBlock-1/conv2d_2/Conv2D:output:03ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@2
ConvBlock-1/conv2d_2/BiasAdd�
ConvBlock-1/activation_2/ReluRelu%ConvBlock-1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������..@2
ConvBlock-1/activation_2/Relu�
0ConvBlock-1/batch_normalization_2/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_2/ReadVariableOp�
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1�
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_2/Relu:activations:08ConvBlock-1/batch_normalization_2/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_2/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������..@:@:@:@:@:*
epsilon%o�:*
is_training( 24
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3�
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02,
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp�
ConvBlock-1/conv2d_3/Conv2DConv2D6ConvBlock-1/batch_normalization_2/FusedBatchNormV3:y:02ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_3/Conv2D�
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp�
ConvBlock-1/conv2d_3/BiasAddBiasAdd$ConvBlock-1/conv2d_3/Conv2D:output:03ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@2
ConvBlock-1/conv2d_3/BiasAdd�
ConvBlock-1/activation_3/ReluRelu%ConvBlock-1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������&&@2
ConvBlock-1/activation_3/Relu�
0ConvBlock-1/batch_normalization_3/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_3/ReadVariableOp�
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1�
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_3/Relu:activations:08ConvBlock-1/batch_normalization_3/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_3/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������&&@:@:@:@:@:*
epsilon%o�:*
is_training( 24
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3�
max_pooling2d_1/MaxPoolMaxPool6ConvBlock-1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool�
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02,
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp�
ConvBlock-2/conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:02ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
ConvBlock-2/conv2d_4/Conv2D�
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp�
ConvBlock-2/conv2d_4/BiasAddBiasAdd$ConvBlock-2/conv2d_4/Conv2D:output:03ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/conv2d_4/BiasAdd�
ConvBlock-2/activation_4/ReluRelu%ConvBlock-2/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/activation_4/Relu�
0ConvBlock-2/batch_normalization_4/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype022
0ConvBlock-2/batch_normalization_4/ReadVariableOp�
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype024
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1�
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02C
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02E
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_4/Relu:activations:08ConvBlock-2/batch_normalization_4/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_4/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 24
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3�
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02,
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp�
ConvBlock-2/conv2d_5/Conv2DConv2D6ConvBlock-2/batch_normalization_4/FusedBatchNormV3:y:02ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
ConvBlock-2/conv2d_5/Conv2D�
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp�
ConvBlock-2/conv2d_5/BiasAddBiasAdd$ConvBlock-2/conv2d_5/Conv2D:output:03ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/conv2d_5/BiasAdd�
ConvBlock-2/activation_5/ReluRelu%ConvBlock-2/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
ConvBlock-2/activation_5/Relu�
0ConvBlock-2/batch_normalization_5/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype022
0ConvBlock-2/batch_normalization_5/ReadVariableOp�
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype024
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1�
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02C
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02E
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_5/Relu:activations:08ConvBlock-2/batch_normalization_5/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_5/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 24
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3�
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices�
global_average_pooling2d/MeanMean6ConvBlock-2/batch_normalization_5/FusedBatchNormV3:y:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
global_average_pooling2d/Mean�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAdd�
!monte_carlo_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2#
!monte_carlo_dropout/dropout/Const�
monte_carlo_dropout/dropout/MulMuldense/BiasAdd:output:0*monte_carlo_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2!
monte_carlo_dropout/dropout/Mul�
!monte_carlo_dropout/dropout/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2#
!monte_carlo_dropout/dropout/Shape�
8monte_carlo_dropout/dropout/random_uniform/RandomUniformRandomUniform*monte_carlo_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"2:
8monte_carlo_dropout/dropout/random_uniform/RandomUniform�
*monte_carlo_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*monte_carlo_dropout/dropout/GreaterEqual/y�
(monte_carlo_dropout/dropout/GreaterEqualGreaterEqualAmonte_carlo_dropout/dropout/random_uniform/RandomUniform:output:03monte_carlo_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2*
(monte_carlo_dropout/dropout/GreaterEqual�
 monte_carlo_dropout/dropout/CastCast,monte_carlo_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2"
 monte_carlo_dropout/dropout/Cast�
!monte_carlo_dropout/dropout/Mul_1Mul#monte_carlo_dropout/dropout/Mul:z:0$monte_carlo_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2#
!monte_carlo_dropout/dropout/Mul_1�
activation_6/ReluRelu%monte_carlo_dropout/dropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2
activation_6/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulactivation_6/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAdd�
#monte_carlo_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#monte_carlo_dropout_1/dropout/Const�
!monte_carlo_dropout_1/dropout/MulMuldense_1/BiasAdd:output:0,monte_carlo_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������2#
!monte_carlo_dropout_1/dropout/Mul�
#monte_carlo_dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2%
#monte_carlo_dropout_1/dropout/Shape�
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniformRandomUniform,monte_carlo_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"*
seed22<
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniform�
,monte_carlo_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,monte_carlo_dropout_1/dropout/GreaterEqual/y�
*monte_carlo_dropout_1/dropout/GreaterEqualGreaterEqualCmonte_carlo_dropout_1/dropout/random_uniform/RandomUniform:output:05monte_carlo_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2,
*monte_carlo_dropout_1/dropout/GreaterEqual�
"monte_carlo_dropout_1/dropout/CastCast.monte_carlo_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2$
"monte_carlo_dropout_1/dropout/Cast�
#monte_carlo_dropout_1/dropout/Mul_1Mul%monte_carlo_dropout_1/dropout/Mul:z:0&monte_carlo_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2%
#monte_carlo_dropout_1/dropout/Mul_1�
activation_7/ReluRelu'monte_carlo_dropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2
activation_7/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMulactivation_7/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_2/Sigmoid�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mulg
IdentityIdentitydense_2/Sigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:::::::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_layer_call_fn_411999

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4119932
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
{
__inference_loss_fn_2_416083J
Fconvblock_0_conv2d_1_kernel_regularizer_square_readvariableop_resource
identity��
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_0_conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulr
IdentityIdentity/ConvBlock-0/conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
|
__inference_loss_fn_10_416171J
Fconvblock_2_conv2d_5_kernel_regularizer_square_readvariableop_resource
identity��
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_2_conv2d_5_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulr
IdentityIdentity/ConvBlock-2/conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
�
4__inference_batch_normalization_layer_call_fn_416257

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_4118412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
{
__inference_loss_fn_4_416105J
Fconvblock_1_conv2d_2_kernel_regularizer_square_readvariableop_resource
identity��
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_1_conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulr
IdentityIdentity/ConvBlock-1/conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
L
0__inference_max_pooling2d_1_layer_call_fn_412219

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4122132
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_412312

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_413330

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_416530

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_415989

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_dense_2_layer_call_fn_416045

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4133902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_3_layer_call_fn_416486

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4121962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412213

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�R
�
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_412945
x+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity��
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@2
conv2d_2/BiasAdd�
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������..@2
activation_2/Relu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������..@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@2
conv2d_3/BiasAdd�
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������&&@2
activation_3/Relu�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������&&@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������66 :::::::::::::R N
/
_output_shapes
:���������66 

_user_specified_namex
��
�
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_413479
input_1
random_rotation_412568
convblock_0_412775
convblock_0_412777
convblock_0_412779
convblock_0_412781
convblock_0_412783
convblock_0_412785
convblock_0_412787
convblock_0_412789
convblock_0_412791
convblock_0_412793
convblock_0_412795
convblock_0_412797
convblock_1_413005
convblock_1_413007
convblock_1_413009
convblock_1_413011
convblock_1_413013
convblock_1_413015
convblock_1_413017
convblock_1_413019
convblock_1_413021
convblock_1_413023
convblock_1_413025
convblock_1_413027
convblock_2_413235
convblock_2_413237
convblock_2_413239
convblock_2_413241
convblock_2_413243
convblock_2_413245
convblock_2_413247
convblock_2_413249
convblock_2_413251
convblock_2_413253
convblock_2_413255
convblock_2_413257
dense_413282
dense_413284
dense_1_413341
dense_1_413343
dense_2_413401
dense_2_413403
identity��#ConvBlock-0/StatefulPartitionedCall�#ConvBlock-1/StatefulPartitionedCall�#ConvBlock-2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+monte_carlo_dropout/StatefulPartitionedCall�-monte_carlo_dropout_1/StatefulPartitionedCall�'random_rotation/StatefulPartitionedCall�
'random_rotation/StatefulPartitionedCallStatefulPartitionedCallinput_1random_rotation_412568*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_4125502)
'random_rotation/StatefulPartitionedCall�
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0convblock_0_412775convblock_0_412777convblock_0_412779convblock_0_412781convblock_0_412783convblock_0_412785convblock_0_412787convblock_0_412789convblock_0_412791convblock_0_412793convblock_0_412795convblock_0_412797*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������ll **
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_4126452%
#ConvBlock-0/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4119932
max_pooling2d/PartitionedCall�
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_413005convblock_1_413007convblock_1_413009convblock_1_413011convblock_1_413013convblock_1_413015convblock_1_413017convblock_1_413019convblock_1_413021convblock_1_413023convblock_1_413025convblock_1_413027*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������&&@**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_4128752%
#ConvBlock-1/StatefulPartitionedCall�
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4122132!
max_pooling2d_1/PartitionedCall�
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_413235convblock_2_413237convblock_2_413239convblock_2_413241convblock_2_413243convblock_2_413245convblock_2_413247convblock_2_413249convblock_2_413251convblock_2_413253convblock_2_413255convblock_2_413257*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_4131052%
#ConvBlock-2/StatefulPartitionedCall�
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4124342*
(global_average_pooling2d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_413282dense_413284*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4132712
dense/StatefulPartitionedCall�
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_4132992-
+monte_carlo_dropout/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_4133122
activation_6/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_413341dense_1_413343*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4133302!
dense_1/StatefulPartitionedCall�
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_4133582/
-monte_carlo_dropout_1/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_4133712
activation_7/PartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_413401dense_2_413403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4133902!
dense_2/StatefulPartitionedCall�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_412775*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_412777*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_412787*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_412789*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413005*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413007*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413017*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_413019*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413235*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413237*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413247*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_413249*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������:::::::::::::::::::::::::::::::::::::::::::2J
#ConvBlock-0/StatefulPartitionedCall#ConvBlock-0/StatefulPartitionedCall2J
#ConvBlock-1/StatefulPartitionedCall#ConvBlock-1/StatefulPartitionedCall2J
#ConvBlock-2/StatefulPartitionedCall#ConvBlock-2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+monte_carlo_dropout/StatefulPartitionedCall+monte_carlo_dropout/StatefulPartitionedCall2^
-monte_carlo_dropout_1/StatefulPartitionedCall-monte_carlo_dropout_1/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_416036

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�6
"__inference__traced_restore_417187
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias
assignvariableop_6_decay$
 assignvariableop_7_learning_rate
assignvariableop_8_momentum
assignvariableop_9_sgd_iter1
-assignvariableop_10_convblock_0_conv2d_kernel/
+assignvariableop_11_convblock_0_conv2d_bias3
/assignvariableop_12_convblock_0_conv2d_1_kernel1
-assignvariableop_13_convblock_0_conv2d_1_bias=
9assignvariableop_14_convblock_0_batch_normalization_gamma<
8assignvariableop_15_convblock_0_batch_normalization_beta?
;assignvariableop_16_convblock_0_batch_normalization_1_gamma>
:assignvariableop_17_convblock_0_batch_normalization_1_betaC
?assignvariableop_18_convblock_0_batch_normalization_moving_meanG
Cassignvariableop_19_convblock_0_batch_normalization_moving_varianceE
Aassignvariableop_20_convblock_0_batch_normalization_1_moving_meanI
Eassignvariableop_21_convblock_0_batch_normalization_1_moving_variance3
/assignvariableop_22_convblock_1_conv2d_2_kernel1
-assignvariableop_23_convblock_1_conv2d_2_bias3
/assignvariableop_24_convblock_1_conv2d_3_kernel1
-assignvariableop_25_convblock_1_conv2d_3_bias?
;assignvariableop_26_convblock_1_batch_normalization_2_gamma>
:assignvariableop_27_convblock_1_batch_normalization_2_beta?
;assignvariableop_28_convblock_1_batch_normalization_3_gamma>
:assignvariableop_29_convblock_1_batch_normalization_3_betaE
Aassignvariableop_30_convblock_1_batch_normalization_2_moving_meanI
Eassignvariableop_31_convblock_1_batch_normalization_2_moving_varianceE
Aassignvariableop_32_convblock_1_batch_normalization_3_moving_meanI
Eassignvariableop_33_convblock_1_batch_normalization_3_moving_variance3
/assignvariableop_34_convblock_2_conv2d_4_kernel1
-assignvariableop_35_convblock_2_conv2d_4_bias3
/assignvariableop_36_convblock_2_conv2d_5_kernel1
-assignvariableop_37_convblock_2_conv2d_5_bias?
;assignvariableop_38_convblock_2_batch_normalization_4_gamma>
:assignvariableop_39_convblock_2_batch_normalization_4_beta?
;assignvariableop_40_convblock_2_batch_normalization_5_gamma>
:assignvariableop_41_convblock_2_batch_normalization_5_betaE
Aassignvariableop_42_convblock_2_batch_normalization_4_moving_meanI
Eassignvariableop_43_convblock_2_batch_normalization_4_moving_varianceE
Aassignvariableop_44_convblock_2_batch_normalization_5_moving_meanI
Eassignvariableop_45_convblock_2_batch_normalization_5_moving_variance 
assignvariableop_46_variable
assignvariableop_47_total
assignvariableop_48_count
assignvariableop_49_total_1
assignvariableop_50_count_1&
"assignvariableop_51_true_positives'
#assignvariableop_52_false_positives(
$assignvariableop_53_true_positives_1'
#assignvariableop_54_false_negatives 
assignvariableop_55_total_cm1
-assignvariableop_56_sgd_dense_kernel_momentum/
+assignvariableop_57_sgd_dense_bias_momentum3
/assignvariableop_58_sgd_dense_1_kernel_momentum1
-assignvariableop_59_sgd_dense_1_bias_momentum3
/assignvariableop_60_sgd_dense_2_kernel_momentum1
-assignvariableop_61_sgd_dense_2_bias_momentum>
:assignvariableop_62_sgd_convblock_0_conv2d_kernel_momentum<
8assignvariableop_63_sgd_convblock_0_conv2d_bias_momentum@
<assignvariableop_64_sgd_convblock_0_conv2d_1_kernel_momentum>
:assignvariableop_65_sgd_convblock_0_conv2d_1_bias_momentumJ
Fassignvariableop_66_sgd_convblock_0_batch_normalization_gamma_momentumI
Eassignvariableop_67_sgd_convblock_0_batch_normalization_beta_momentumL
Hassignvariableop_68_sgd_convblock_0_batch_normalization_1_gamma_momentumK
Gassignvariableop_69_sgd_convblock_0_batch_normalization_1_beta_momentum@
<assignvariableop_70_sgd_convblock_1_conv2d_2_kernel_momentum>
:assignvariableop_71_sgd_convblock_1_conv2d_2_bias_momentum@
<assignvariableop_72_sgd_convblock_1_conv2d_3_kernel_momentum>
:assignvariableop_73_sgd_convblock_1_conv2d_3_bias_momentumL
Hassignvariableop_74_sgd_convblock_1_batch_normalization_2_gamma_momentumK
Gassignvariableop_75_sgd_convblock_1_batch_normalization_2_beta_momentumL
Hassignvariableop_76_sgd_convblock_1_batch_normalization_3_gamma_momentumK
Gassignvariableop_77_sgd_convblock_1_batch_normalization_3_beta_momentum@
<assignvariableop_78_sgd_convblock_2_conv2d_4_kernel_momentum>
:assignvariableop_79_sgd_convblock_2_conv2d_4_bias_momentum@
<assignvariableop_80_sgd_convblock_2_conv2d_5_kernel_momentum>
:assignvariableop_81_sgd_convblock_2_conv2d_5_bias_momentumL
Hassignvariableop_82_sgd_convblock_2_batch_normalization_4_gamma_momentumK
Gassignvariableop_83_sgd_convblock_2_batch_normalization_4_beta_momentumL
Hassignvariableop_84_sgd_convblock_2_batch_normalization_5_gamma_momentumK
Gassignvariableop_85_sgd_convblock_2_batch_normalization_5_beta_momentum
identity_87��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_9�'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*�&
value�&B�&WB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB2layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB7keras_api/metrics/4/total_cm/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*�
value�B�WB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*e
dtypes[
Y2W		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_convblock_0_conv2d_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp+assignvariableop_11_convblock_0_conv2d_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_convblock_0_conv2d_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp-assignvariableop_13_convblock_0_conv2d_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp9assignvariableop_14_convblock_0_batch_normalization_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp8assignvariableop_15_convblock_0_batch_normalization_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp;assignvariableop_16_convblock_0_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_convblock_0_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp?assignvariableop_18_convblock_0_batch_normalization_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpCassignvariableop_19_convblock_0_batch_normalization_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpAassignvariableop_20_convblock_0_batch_normalization_1_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpEassignvariableop_21_convblock_0_batch_normalization_1_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp/assignvariableop_22_convblock_1_conv2d_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp-assignvariableop_23_convblock_1_conv2d_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_convblock_1_conv2d_3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp-assignvariableop_25_convblock_1_conv2d_3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp;assignvariableop_26_convblock_1_batch_normalization_2_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp:assignvariableop_27_convblock_1_batch_normalization_2_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp;assignvariableop_28_convblock_1_batch_normalization_3_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp:assignvariableop_29_convblock_1_batch_normalization_3_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpAassignvariableop_30_convblock_1_batch_normalization_2_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpEassignvariableop_31_convblock_1_batch_normalization_2_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOpAassignvariableop_32_convblock_1_batch_normalization_3_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpEassignvariableop_33_convblock_1_batch_normalization_3_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp/assignvariableop_34_convblock_2_conv2d_4_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp-assignvariableop_35_convblock_2_conv2d_4_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_convblock_2_conv2d_5_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp-assignvariableop_37_convblock_2_conv2d_5_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp;assignvariableop_38_convblock_2_batch_normalization_4_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp:assignvariableop_39_convblock_2_batch_normalization_4_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp;assignvariableop_40_convblock_2_batch_normalization_5_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp:assignvariableop_41_convblock_2_batch_normalization_5_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOpAassignvariableop_42_convblock_2_batch_normalization_4_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOpEassignvariableop_43_convblock_2_batch_normalization_4_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOpAassignvariableop_44_convblock_2_batch_normalization_5_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOpEassignvariableop_45_convblock_2_batch_normalization_5_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOpassignvariableop_46_variableIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_1Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp"assignvariableop_51_true_positivesIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp#assignvariableop_52_false_positivesIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp$assignvariableop_53_true_positives_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp#assignvariableop_54_false_negativesIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_cmIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp-assignvariableop_56_sgd_dense_kernel_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp+assignvariableop_57_sgd_dense_bias_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp/assignvariableop_58_sgd_dense_1_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp-assignvariableop_59_sgd_dense_1_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp/assignvariableop_60_sgd_dense_2_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp-assignvariableop_61_sgd_dense_2_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp:assignvariableop_62_sgd_convblock_0_conv2d_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp8assignvariableop_63_sgd_convblock_0_conv2d_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp<assignvariableop_64_sgd_convblock_0_conv2d_1_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp:assignvariableop_65_sgd_convblock_0_conv2d_1_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOpFassignvariableop_66_sgd_convblock_0_batch_normalization_gamma_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOpEassignvariableop_67_sgd_convblock_0_batch_normalization_beta_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOpHassignvariableop_68_sgd_convblock_0_batch_normalization_1_gamma_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOpGassignvariableop_69_sgd_convblock_0_batch_normalization_1_beta_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp<assignvariableop_70_sgd_convblock_1_conv2d_2_kernel_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp:assignvariableop_71_sgd_convblock_1_conv2d_2_bias_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp<assignvariableop_72_sgd_convblock_1_conv2d_3_kernel_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp:assignvariableop_73_sgd_convblock_1_conv2d_3_bias_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOpHassignvariableop_74_sgd_convblock_1_batch_normalization_2_gamma_momentumIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOpGassignvariableop_75_sgd_convblock_1_batch_normalization_2_beta_momentumIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOpHassignvariableop_76_sgd_convblock_1_batch_normalization_3_gamma_momentumIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOpGassignvariableop_77_sgd_convblock_1_batch_normalization_3_beta_momentumIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp<assignvariableop_78_sgd_convblock_2_conv2d_4_kernel_momentumIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp:assignvariableop_79_sgd_convblock_2_conv2d_4_bias_momentumIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp<assignvariableop_80_sgd_convblock_2_conv2d_5_kernel_momentumIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp:assignvariableop_81_sgd_convblock_2_conv2d_5_bias_momentumIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOpHassignvariableop_82_sgd_convblock_2_batch_normalization_4_gamma_momentumIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOpGassignvariableop_83_sgd_convblock_2_batch_normalization_4_beta_momentumIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84�
AssignVariableOp_84AssignVariableOpHassignvariableop_84_sgd_convblock_2_batch_normalization_5_gamma_momentumIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85�
AssignVariableOp_85AssignVariableOpGassignvariableop_85_sgd_convblock_2_batch_normalization_5_beta_momentumIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_859
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_86Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_86�
Identity_87IdentityIdentity_86:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_87"#
identity_87Identity_87:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_412092

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
m
4__inference_monte_carlo_dropout_layer_call_fn_415969

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_4132992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�S
�
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_413175
x+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity��
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_4/BiasAdd�
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_4/Relu�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2d_5/Conv2D�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_5/BiasAdd�
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
activation_5/Relu�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&batch_normalization_5/ReadVariableOp_1�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3�
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mul�
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:��20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square�
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Const�
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum�
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x�
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mul�
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������@:::::::::::::R N
/
_output_shapes
:���������@

_user_specified_namex
�d
�
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_415579
x+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity��$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�$batch_normalization_3/AssignNewValue�&batch_normalization_3/AssignNewValue_1�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@*
paddingVALID*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������..@2
conv2d_2/BiasAdd�
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������..@2
activation_2/Relu�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������..@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_2/FusedBatchNormV3�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������&&@2
conv2d_3/BiasAdd�
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������&&@2
activation_3/Relu�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������&&@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2(
&batch_normalization_3/FusedBatchNormV3�
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue�
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1�
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square�
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Const�
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mul�
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square�
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Const�
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum�
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x�
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mul�
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square�
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Const�
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum�
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/x�
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:���������&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������66 ::::::::::::2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:R N
/
_output_shapes
:���������66 

_user_specified_namex
�
y
__inference_loss_fn_9_416160H
Dconvblock_2_conv2d_4_bias_regularizer_square_readvariableop_resource
identity��
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_2_conv2d_4_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square�
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Const�
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulp
IdentityIdentity-ConvBlock-2/conv2d_4/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
}
(__inference_dense_1_layer_call_fn_415998

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4133302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_416244

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
z
__inference_loss_fn_11_416182H
Dconvblock_2_conv2d_5_bias_regularizer_square_readvariableop_resource
identity��
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_2_conv2d_5_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:�*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square�
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Const�
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum�
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/x�
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mulp
IdentityIdentity-ConvBlock-2/conv2d_5/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_411841

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
g
K__inference_random_rotation_layer_call_and_return_conditional_losses_412554

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
,__inference_ConvBlock-2_layer_call_fn_415904
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_4131752
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������@

_user_specified_namex
�
�
A__inference_dense_layer_call_and_return_conditional_losses_413271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
,__inference_ConvBlock-0_layer_call_fn_415452
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������ll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_4127152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������ll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:�����������

_user_specified_namex
�
L
0__inference_random_rotation_layer_call_fn_415255

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_4125542
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
n
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_413299

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*
seed�"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_416460

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
d
H__inference_activation_7_layer_call_and_return_conditional_losses_416020

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_412061

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_416226

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��AssignNewValue�AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<2
FusedBatchNormV3�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1�
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_415943

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
w
__inference_loss_fn_1_416072F
Bconvblock_0_conv2d_bias_regularizer_square_readvariableop_resource
identity��
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpBconvblock_0_conv2d_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/muln
IdentityIdentity+ConvBlock-0/conv2d/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�
d
H__inference_activation_6_layer_call_and_return_conditional_losses_413312

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
,__inference_ConvBlock-1_layer_call_fn_415678
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_4129452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:���������66 ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������66 

_user_specified_namex
�Q
�
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_412715
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity��
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv *
paddingVALID*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������vv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������vv 2
activation/Relu�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������vv : : : : :*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll *
paddingVALID*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ll 2
conv2d_1/BiasAdd�
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������ll 2
activation_1/Relu�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������ll : : : : :*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3�
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square�
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Const�
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum�
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/x�
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mul�
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp�
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square�
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/Const�
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum�
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/x�
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mul�
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp�
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square�
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Const�
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum�
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x�
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mul�
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp�
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square�
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Const�
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum�
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *���=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/x�
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul�
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������ll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:�����������:::::::::::::T P
1
_output_shapes
:�����������

_user_specified_namex
�
�
3__inference_FERREIRA2020_class_layer_call_fn_414184
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_4140972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:�����������::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������;
dense_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�0
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer-13
layer_with_weights-5
layer-14
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�,
_tf_keras_network�,{"class_name": "Functional", "name": "FERREIRA2020_class", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "FERREIRA2020_class", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": {"class_name": "__tuple__", "items": [0.15, 0.15]}, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}, "name": "random_rotation", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "ConvBlock", "config": {"layer was saved without config": true}, "name": "ConvBlock-0", "inbound_nodes": [[["random_rotation", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["ConvBlock-0", 0, 0, {}]]]}, {"class_name": "ConvBlock", "config": {"layer was saved without config": true}, "name": "ConvBlock-1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["ConvBlock-1", 0, 0, {}]]]}, {"class_name": "ConvBlock", "config": {"layer was saved without config": true}, "name": "ConvBlock-2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["ConvBlock-2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "MonteCarloDropout", "config": {"name": "monte_carlo_dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "monte_carlo_dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["monte_carlo_dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "MonteCarloDropout", "config": {"name": "monte_carlo_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "monte_carlo_dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["monte_carlo_dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": ["accuracy", {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Upsilon", "config": {"name": "upsilon", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 6.155829851195449e-06, "decay": 9.999999747378752e-05, "momentum": 0.8999999761581421, "nesterov": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�
_rng
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "RandomRotation", "name": "random_rotation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": {"class_name": "__tuple__", "items": [0.15, 0.15]}, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}
�

conv2d

activation

batch_norm
regularization_losses
	variables
 trainable_variables
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ConvBlock", "name": "ConvBlock-0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

&conv2d
'
activation
(
batch_norm
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ConvBlock", "name": "ConvBlock-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

1conv2d
2
activation
3
batch_norm
4regularization_losses
5	variables
6trainable_variables
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "ConvBlock", "name": "ConvBlock-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
8regularization_losses
9	variables
:trainable_variables
;	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MonteCarloDropout", "name": "monte_carlo_dropout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "monte_carlo_dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}
�

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MonteCarloDropout", "name": "monte_carlo_dropout_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "monte_carlo_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}}
�

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�
	^decay
_learning_rate
`momentum
aiter<momentum�=momentum�Jmomentum�Kmomentum�Xmomentum�Ymomentum�bmomentum�cmomentum�dmomentum�emomentum�fmomentum�gmomentum�hmomentum�imomentum�nmomentum�omomentum�pmomentum�qmomentum�rmomentum�smomentum�tmomentum�umomentum�zmomentum�{momentum�|momentum�}momentum�~momentum�momentum��momentum��momentum�"
	optimizer
 "
trackable_list_wrapper
�
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11
n12
o13
p14
q15
r16
s17
t18
u19
v20
w21
x22
y23
z24
{25
|26
}27
~28
29
�30
�31
�32
�33
�34
�35
<36
=37
J38
K39
X40
Y41"
trackable_list_wrapper
�
b0
c1
d2
e3
f4
g5
h6
i7
n8
o9
p10
q11
r12
s13
t14
u15
z16
{17
|18
}19
~20
21
�22
�23
<24
=25
J26
K27
X28
Y29"
trackable_list_wrapper
�
�layer_metrics
regularization_losses
�non_trainable_variables
�metrics
	variables
�layers
trainable_variables
 �layer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
/
�
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
regularization_losses
�metrics
	variables
�layers
trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
v
b0
c1
d2
e3
f4
g5
h6
i7
j8
k9
l10
m11"
trackable_list_wrapper
X
b0
c1
d2
e3
f4
g5
h6
i7"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
regularization_losses
�metrics
	variables
�layers
 trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
"regularization_losses
�metrics
#	variables
�layers
$trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
v
n0
o1
p2
q3
r4
s5
t6
u7
v8
w9
x10
y11"
trackable_list_wrapper
X
n0
o1
p2
q3
r4
s5
t6
u7"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
)regularization_losses
�metrics
*	variables
�layers
+trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
-regularization_losses
�metrics
.	variables
�layers
/trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
|
z0
{1
|2
}3
~4
5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
Z
z0
{1
|2
}3
~4
5
�6
�7"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
4regularization_losses
�metrics
5	variables
�layers
6trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
8regularization_losses
�metrics
9	variables
�layers
:trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :
��2dense/kernel
:�2
dense/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
>regularization_losses
�metrics
?	variables
�layers
@trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Bregularization_losses
�metrics
C	variables
�layers
Dtrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Fregularization_losses
�metrics
G	variables
�layers
Htrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 
��2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Lregularization_losses
�metrics
M	variables
�layers
Ntrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Pregularization_losses
�metrics
Q	variables
�layers
Rtrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Tregularization_losses
�metrics
U	variables
�layers
Vtrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
Zregularization_losses
�metrics
[	variables
�layers
\trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
3:1 2ConvBlock-0/conv2d/kernel
%:# 2ConvBlock-0/conv2d/bias
5:3  2ConvBlock-0/conv2d_1/kernel
':% 2ConvBlock-0/conv2d_1/bias
3:1 2%ConvBlock-0/batch_normalization/gamma
2:0 2$ConvBlock-0/batch_normalization/beta
5:3 2'ConvBlock-0/batch_normalization_1/gamma
4:2 2&ConvBlock-0/batch_normalization_1/beta
;:9  (2+ConvBlock-0/batch_normalization/moving_mean
?:=  (2/ConvBlock-0/batch_normalization/moving_variance
=:;  (2-ConvBlock-0/batch_normalization_1/moving_mean
A:?  (21ConvBlock-0/batch_normalization_1/moving_variance
5:3		 @2ConvBlock-1/conv2d_2/kernel
':%@2ConvBlock-1/conv2d_2/bias
5:3		@@2ConvBlock-1/conv2d_3/kernel
':%@2ConvBlock-1/conv2d_3/bias
5:3@2'ConvBlock-1/batch_normalization_2/gamma
4:2@2&ConvBlock-1/batch_normalization_2/beta
5:3@2'ConvBlock-1/batch_normalization_3/gamma
4:2@2&ConvBlock-1/batch_normalization_3/beta
=:;@ (2-ConvBlock-1/batch_normalization_2/moving_mean
A:?@ (21ConvBlock-1/batch_normalization_2/moving_variance
=:;@ (2-ConvBlock-1/batch_normalization_3/moving_mean
A:?@ (21ConvBlock-1/batch_normalization_3/moving_variance
6:4@�2ConvBlock-2/conv2d_4/kernel
(:&�2ConvBlock-2/conv2d_4/bias
7:5��2ConvBlock-2/conv2d_5/kernel
(:&�2ConvBlock-2/conv2d_5/bias
6:4�2'ConvBlock-2/batch_normalization_4/gamma
5:3�2&ConvBlock-2/batch_normalization_4/beta
6:4�2'ConvBlock-2/batch_normalization_5/gamma
5:3�2&ConvBlock-2/batch_normalization_5/beta
>:<� (2-ConvBlock-2/batch_normalization_4/moving_mean
B:@� (21ConvBlock-2/batch_normalization_4/moving_variance
>:<� (2-ConvBlock-2/batch_normalization_5/moving_mean
B:@� (21ConvBlock-2/batch_normalization_5/moving_variance
 "
trackable_dict_wrapper
z
j0
k1
l2
m3
v4
w5
x6
y7
�8
�9
�10
�11"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


bkernel
cbias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
�


dkernel
ebias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 118, 118, 32]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
�	
	�axis
	fgamma
gbeta
jmoving_mean
kmoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 118, 118, 32]}}
�	
	�axis
	hgamma
ibeta
lmoving_mean
mmoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 108, 108, 32]}}
 "
trackable_dict_wrapper
<
j0
k1
l2
m3"
trackable_list_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


nkernel
obias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 54, 54, 32]}}
�


pkernel
qbias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 64]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
�	
	�axis
	rgamma
sbeta
vmoving_mean
wmoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 64]}}
�	
	�axis
	tgamma
ubeta
xmoving_mean
ymoving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 38, 38, 64]}}
 "
trackable_dict_wrapper
<
v0
w1
x2
y3"
trackable_list_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


zkernel
{bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 64]}}
�


|kernel
}bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 128]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
�	
	�axis
	~gamma
beta
�moving_mean
�moving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 128]}}
�	
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 128]}}
 "
trackable_dict_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
�
�
thresholds
�true_positives
�false_positives
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
�
�
thresholds
�true_positives
�false_negatives
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
�
�total_cm
�	variables
�	keras_api"�
_tf_keras_metrics{"class_name": "Upsilon", "name": "upsilon", "dtype": "float32", "config": {"name": "upsilon", "dtype": "float32"}}
0
�0
�1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
f0
g1
j2
k3"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
h0
i1
l2
m3"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
r0
s1
v2
w3"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
t0
u1
x2
y3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
>
~0
1
�2
�3"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�layer_metrics
�non_trainable_variables
�regularization_losses
�metrics
�	variables
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
: (2total_cm
(
�0"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
+:)
��2SGD/dense/kernel/momentum
$:"�2SGD/dense/bias/momentum
-:+
��2SGD/dense_1/kernel/momentum
&:$�2SGD/dense_1/bias/momentum
,:*	�2SGD/dense_2/kernel/momentum
%:#2SGD/dense_2/bias/momentum
>:< 2&SGD/ConvBlock-0/conv2d/kernel/momentum
0:. 2$SGD/ConvBlock-0/conv2d/bias/momentum
@:>  2(SGD/ConvBlock-0/conv2d_1/kernel/momentum
2:0 2&SGD/ConvBlock-0/conv2d_1/bias/momentum
>:< 22SGD/ConvBlock-0/batch_normalization/gamma/momentum
=:; 21SGD/ConvBlock-0/batch_normalization/beta/momentum
@:> 24SGD/ConvBlock-0/batch_normalization_1/gamma/momentum
?:= 23SGD/ConvBlock-0/batch_normalization_1/beta/momentum
@:>		 @2(SGD/ConvBlock-1/conv2d_2/kernel/momentum
2:0@2&SGD/ConvBlock-1/conv2d_2/bias/momentum
@:>		@@2(SGD/ConvBlock-1/conv2d_3/kernel/momentum
2:0@2&SGD/ConvBlock-1/conv2d_3/bias/momentum
@:>@24SGD/ConvBlock-1/batch_normalization_2/gamma/momentum
?:=@23SGD/ConvBlock-1/batch_normalization_2/beta/momentum
@:>@24SGD/ConvBlock-1/batch_normalization_3/gamma/momentum
?:=@23SGD/ConvBlock-1/batch_normalization_3/beta/momentum
A:?@�2(SGD/ConvBlock-2/conv2d_4/kernel/momentum
3:1�2&SGD/ConvBlock-2/conv2d_4/bias/momentum
B:@��2(SGD/ConvBlock-2/conv2d_5/kernel/momentum
3:1�2&SGD/ConvBlock-2/conv2d_5/bias/momentum
A:?�24SGD/ConvBlock-2/batch_normalization_4/gamma/momentum
@:>�23SGD/ConvBlock-2/batch_normalization_4/beta/momentum
A:?�24SGD/ConvBlock-2/batch_normalization_5/gamma/momentum
@:>�23SGD/ConvBlock-2/batch_normalization_5/beta/momentum
�2�
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_414710
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_413653
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_414953
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_413479�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
3__inference_FERREIRA2020_class_layer_call_fn_413921
3__inference_FERREIRA2020_class_layer_call_fn_415133
3__inference_FERREIRA2020_class_layer_call_fn_414184
3__inference_FERREIRA2020_class_layer_call_fn_415044�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_411779�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������
�2�
K__inference_random_rotation_layer_call_and_return_conditional_losses_415243
K__inference_random_rotation_layer_call_and_return_conditional_losses_415239�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_random_rotation_layer_call_fn_415250
0__inference_random_rotation_layer_call_fn_415255�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_415353
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_415423�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_ConvBlock-0_layer_call_fn_415481
,__inference_ConvBlock-0_layer_call_fn_415452�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_411993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_layer_call_fn_411999�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_415649
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_415579�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_ConvBlock-1_layer_call_fn_415707
,__inference_ConvBlock-1_layer_call_fn_415678�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412213�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
0__inference_max_pooling2d_1_layer_call_fn_412219�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_415875
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_415805�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_ConvBlock-2_layer_call_fn_415933
,__inference_ConvBlock-2_layer_call_fn_415904�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_412434�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
9__inference_global_average_pooling2d_layer_call_fn_412440�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
A__inference_dense_layer_call_and_return_conditional_losses_415943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_415952�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_415964�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_monte_carlo_dropout_layer_call_fn_415969�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_6_layer_call_and_return_conditional_losses_415974�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_activation_6_layer_call_fn_415979�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_415989�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_415998�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_416010�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_monte_carlo_dropout_1_layer_call_fn_416015�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_activation_7_layer_call_and_return_conditional_losses_416020�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_activation_7_layer_call_fn_416025�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_416036�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_416045�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
3B1
$__inference_signature_wrapper_414353input_1
�2�
__inference_loss_fn_0_416061�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_416072�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_416083�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_416094�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_416105�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_5_416116�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_6_416127�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_7_416138�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_8_416149�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_9_416160�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_10_416171�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_11_416182�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_416244
O__inference_batch_normalization_layer_call_and_return_conditional_losses_416226�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_batch_normalization_layer_call_fn_416270
4__inference_batch_normalization_layer_call_fn_416257�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_416308
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_416290�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_1_layer_call_fn_416334
6__inference_batch_normalization_1_layer_call_fn_416321�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_416396
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_416378�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_2_layer_call_fn_416422
6__inference_batch_normalization_2_layer_call_fn_416409�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_416442
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_416460�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_3_layer_call_fn_416486
6__inference_batch_normalization_3_layer_call_fn_416473�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_416530
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_416548�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_4_layer_call_fn_416574
6__inference_batch_normalization_4_layer_call_fn_416561�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_416594
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_416612�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_5_layer_call_fn_416625
6__inference_batch_normalization_5_layer_call_fn_416638�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_415353wbcfgjkdehilm8�5
.�+
%�"
x�����������
p
� "-�*
#� 
0���������ll 
� �
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_415423wbcfgjkdehilm8�5
.�+
%�"
x�����������
p 
� "-�*
#� 
0���������ll 
� �
,__inference_ConvBlock-0_layer_call_fn_415452jbcfgjkdehilm8�5
.�+
%�"
x�����������
p
� " ����������ll �
,__inference_ConvBlock-0_layer_call_fn_415481jbcfgjkdehilm8�5
.�+
%�"
x�����������
p 
� " ����������ll �
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_415579unorsvwpqtuxy6�3
,�)
#� 
x���������66 
p
� "-�*
#� 
0���������&&@
� �
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_415649unorsvwpqtuxy6�3
,�)
#� 
x���������66 
p 
� "-�*
#� 
0���������&&@
� �
,__inference_ConvBlock-1_layer_call_fn_415678hnorsvwpqtuxy6�3
,�)
#� 
x���������66 
p
� " ����������&&@�
,__inference_ConvBlock-1_layer_call_fn_415707hnorsvwpqtuxy6�3
,�)
#� 
x���������66 
p 
� " ����������&&@�
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_415805|z{~��|}����6�3
,�)
#� 
x���������@
p
� ".�+
$�!
0����������
� �
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_415875|z{~��|}����6�3
,�)
#� 
x���������@
p 
� ".�+
$�!
0����������
� �
,__inference_ConvBlock-2_layer_call_fn_415904oz{~��|}����6�3
,�)
#� 
x���������@
p
� "!������������
,__inference_ConvBlock-2_layer_call_fn_415933oz{~��|}����6�3
,�)
#� 
x���������@
p 
� "!������������
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_413479�2�bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYB�?
8�5
+�(
input_1�����������
p

 
� "%�"
�
0���������
� �
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_413653�0bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYB�?
8�5
+�(
input_1�����������
p 

 
� "%�"
�
0���������
� �
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_414710�2�bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYA�>
7�4
*�'
inputs�����������
p

 
� "%�"
�
0���������
� �
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_414953�0bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYA�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
3__inference_FERREIRA2020_class_layer_call_fn_413921�2�bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYB�?
8�5
+�(
input_1�����������
p

 
� "�����������
3__inference_FERREIRA2020_class_layer_call_fn_414184�0bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYB�?
8�5
+�(
input_1�����������
p 

 
� "�����������
3__inference_FERREIRA2020_class_layer_call_fn_415044�2�bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYA�>
7�4
*�'
inputs�����������
p

 
� "�����������
3__inference_FERREIRA2020_class_layer_call_fn_415133�0bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYA�>
7�4
*�'
inputs�����������
p 

 
� "�����������
!__inference__wrapped_model_411779�0bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXY:�7
0�-
+�(
input_1�����������
� "1�.
,
dense_2!�
dense_2����������
H__inference_activation_6_layer_call_and_return_conditional_losses_415974Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
-__inference_activation_6_layer_call_fn_415979M0�-
&�#
!�
inputs����������
� "������������
H__inference_activation_7_layer_call_and_return_conditional_losses_416020Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� ~
-__inference_activation_7_layer_call_fn_416025M0�-
&�#
!�
inputs����������
� "������������
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_416290�hilmM�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_416308�hilmM�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
6__inference_batch_normalization_1_layer_call_fn_416321�hilmM�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
6__inference_batch_normalization_1_layer_call_fn_416334�hilmM�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_416378�rsvwM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_416396�rsvwM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
6__inference_batch_normalization_2_layer_call_fn_416409�rsvwM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
6__inference_batch_normalization_2_layer_call_fn_416422�rsvwM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_416442�tuxyM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_416460�tuxyM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
6__inference_batch_normalization_3_layer_call_fn_416473�tuxyM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
6__inference_batch_normalization_3_layer_call_fn_416486�tuxyM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_416530�~��N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_416548�~��N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
6__inference_batch_normalization_4_layer_call_fn_416561�~��N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
6__inference_batch_normalization_4_layer_call_fn_416574�~��N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_416594�����N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_416612�����N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
6__inference_batch_normalization_5_layer_call_fn_416625�����N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
6__inference_batch_normalization_5_layer_call_fn_416638�����N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_416226�fgjkM�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_416244�fgjkM�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
4__inference_batch_normalization_layer_call_fn_416257�fgjkM�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
4__inference_batch_normalization_layer_call_fn_416270�fgjkM�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
C__inference_dense_1_layer_call_and_return_conditional_losses_415989^JK0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_dense_1_layer_call_fn_415998QJK0�-
&�#
!�
inputs����������
� "������������
C__inference_dense_2_layer_call_and_return_conditional_losses_416036]XY0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
(__inference_dense_2_layer_call_fn_416045PXY0�-
&�#
!�
inputs����������
� "�����������
A__inference_dense_layer_call_and_return_conditional_losses_415943^<=0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� {
&__inference_dense_layer_call_fn_415952Q<=0�-
&�#
!�
inputs����������
� "������������
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_412434�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
9__inference_global_average_pooling2d_layer_call_fn_412440wR�O
H�E
C�@
inputs4������������������������������������
� "!�������������������;
__inference_loss_fn_0_416061b�

� 
� "� <
__inference_loss_fn_10_416171|�

� 
� "� <
__inference_loss_fn_11_416182}�

� 
� "� ;
__inference_loss_fn_1_416072c�

� 
� "� ;
__inference_loss_fn_2_416083d�

� 
� "� ;
__inference_loss_fn_3_416094e�

� 
� "� ;
__inference_loss_fn_4_416105n�

� 
� "� ;
__inference_loss_fn_5_416116o�

� 
� "� ;
__inference_loss_fn_6_416127p�

� 
� "� ;
__inference_loss_fn_7_416138q�

� 
� "� ;
__inference_loss_fn_8_416149z�

� 
� "� ;
__inference_loss_fn_9_416160{�

� 
� "� �
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_412213�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_max_pooling2d_1_layer_call_fn_412219�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_411993�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_layer_call_fn_411999�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_416010Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
6__inference_monte_carlo_dropout_1_layer_call_fn_416015M0�-
&�#
!�
inputs����������
� "������������
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_415964Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
4__inference_monte_carlo_dropout_layer_call_fn_415969M0�-
&�#
!�
inputs����������
� "������������
K__inference_random_rotation_layer_call_and_return_conditional_losses_415239t�=�:
3�0
*�'
inputs�����������
p
� "/�,
%�"
0�����������
� �
K__inference_random_rotation_layer_call_and_return_conditional_losses_415243p=�:
3�0
*�'
inputs�����������
p 
� "/�,
%�"
0�����������
� �
0__inference_random_rotation_layer_call_fn_415250g�=�:
3�0
*�'
inputs�����������
p
� ""�������������
0__inference_random_rotation_layer_call_fn_415255c=�:
3�0
*�'
inputs�����������
p 
� ""�������������
$__inference_signature_wrapper_414353�0bcfgjkdehilmnorsvwpqtuxyz{~��|}����<=JKXYE�B
� 
;�8
6
input_1+�(
input_1�����������"1�.
,
dense_2!�
dense_2���������