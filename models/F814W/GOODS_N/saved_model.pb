ª÷.
Î£
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
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.22v2.3.1-38-g9edbe5075f78¦ú(
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
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

ConvBlock-0/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameConvBlock-0/conv2d/kernel

-ConvBlock-0/conv2d/kernel/Read/ReadVariableOpReadVariableOpConvBlock-0/conv2d/kernel*&
_output_shapes
: *
dtype0

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

ConvBlock-0/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameConvBlock-0/conv2d_1/kernel

/ConvBlock-0/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpConvBlock-0/conv2d_1/kernel*&
_output_shapes
:  *
dtype0

ConvBlock-0/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameConvBlock-0/conv2d_1/bias

-ConvBlock-0/conv2d_1/bias/Read/ReadVariableOpReadVariableOpConvBlock-0/conv2d_1/bias*
_output_shapes
: *
dtype0
¢
%ConvBlock-0/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%ConvBlock-0/batch_normalization/gamma

9ConvBlock-0/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp%ConvBlock-0/batch_normalization/gamma*
_output_shapes
: *
dtype0
 
$ConvBlock-0/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$ConvBlock-0/batch_normalization/beta

8ConvBlock-0/batch_normalization/beta/Read/ReadVariableOpReadVariableOp$ConvBlock-0/batch_normalization/beta*
_output_shapes
: *
dtype0
¦
'ConvBlock-0/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'ConvBlock-0/batch_normalization_1/gamma

;ConvBlock-0/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-0/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
¤
&ConvBlock-0/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&ConvBlock-0/batch_normalization_1/beta

:ConvBlock-0/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-0/batch_normalization_1/beta*
_output_shapes
: *
dtype0
®
+ConvBlock-0/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+ConvBlock-0/batch_normalization/moving_mean
§
?ConvBlock-0/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp+ConvBlock-0/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
¶
/ConvBlock-0/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/ConvBlock-0/batch_normalization/moving_variance
¯
CConvBlock-0/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp/ConvBlock-0/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
²
-ConvBlock-0/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-ConvBlock-0/batch_normalization_1/moving_mean
«
AConvBlock-0/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-0/batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
º
1ConvBlock-0/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31ConvBlock-0/batch_normalization_1/moving_variance
³
EConvBlock-0/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-0/batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0

ConvBlock-1/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 @*,
shared_nameConvBlock-1/conv2d_2/kernel

/ConvBlock-1/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_2/kernel*&
_output_shapes
:		 @*
dtype0

ConvBlock-1/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameConvBlock-1/conv2d_2/bias

-ConvBlock-1/conv2d_2/bias/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_2/bias*
_output_shapes
:@*
dtype0

ConvBlock-1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@@*,
shared_nameConvBlock-1/conv2d_3/kernel

/ConvBlock-1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_3/kernel*&
_output_shapes
:		@@*
dtype0

ConvBlock-1/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameConvBlock-1/conv2d_3/bias

-ConvBlock-1/conv2d_3/bias/Read/ReadVariableOpReadVariableOpConvBlock-1/conv2d_3/bias*
_output_shapes
:@*
dtype0
¦
'ConvBlock-1/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'ConvBlock-1/batch_normalization_2/gamma

;ConvBlock-1/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-1/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
¤
&ConvBlock-1/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&ConvBlock-1/batch_normalization_2/beta

:ConvBlock-1/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-1/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
¦
'ConvBlock-1/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'ConvBlock-1/batch_normalization_3/gamma

;ConvBlock-1/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-1/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
¤
&ConvBlock-1/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&ConvBlock-1/batch_normalization_3/beta

:ConvBlock-1/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-1/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
²
-ConvBlock-1/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-ConvBlock-1/batch_normalization_2/moving_mean
«
AConvBlock-1/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-1/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
º
1ConvBlock-1/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31ConvBlock-1/batch_normalization_2/moving_variance
³
EConvBlock-1/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-1/batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
²
-ConvBlock-1/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-ConvBlock-1/batch_normalization_3/moving_mean
«
AConvBlock-1/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-1/batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
º
1ConvBlock-1/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31ConvBlock-1/batch_normalization_3/moving_variance
³
EConvBlock-1/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-1/batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0

ConvBlock-2/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameConvBlock-2/conv2d_4/kernel

/ConvBlock-2/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_4/kernel*'
_output_shapes
:@*
dtype0

ConvBlock-2/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameConvBlock-2/conv2d_4/bias

-ConvBlock-2/conv2d_4/bias/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_4/bias*
_output_shapes	
:*
dtype0

ConvBlock-2/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameConvBlock-2/conv2d_5/kernel

/ConvBlock-2/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_5/kernel*(
_output_shapes
:*
dtype0

ConvBlock-2/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameConvBlock-2/conv2d_5/bias

-ConvBlock-2/conv2d_5/bias/Read/ReadVariableOpReadVariableOpConvBlock-2/conv2d_5/bias*
_output_shapes	
:*
dtype0
§
'ConvBlock-2/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'ConvBlock-2/batch_normalization_4/gamma
 
;ConvBlock-2/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-2/batch_normalization_4/gamma*
_output_shapes	
:*
dtype0
¥
&ConvBlock-2/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&ConvBlock-2/batch_normalization_4/beta

:ConvBlock-2/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-2/batch_normalization_4/beta*
_output_shapes	
:*
dtype0
§
'ConvBlock-2/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'ConvBlock-2/batch_normalization_5/gamma
 
;ConvBlock-2/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp'ConvBlock-2/batch_normalization_5/gamma*
_output_shapes	
:*
dtype0
¥
&ConvBlock-2/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&ConvBlock-2/batch_normalization_5/beta

:ConvBlock-2/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp&ConvBlock-2/batch_normalization_5/beta*
_output_shapes	
:*
dtype0
³
-ConvBlock-2/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-ConvBlock-2/batch_normalization_4/moving_mean
¬
AConvBlock-2/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-2/batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
»
1ConvBlock-2/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31ConvBlock-2/batch_normalization_4/moving_variance
´
EConvBlock-2/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-2/batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0
³
-ConvBlock-2/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-ConvBlock-2/batch_normalization_5/moving_mean
¬
AConvBlock-2/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp-ConvBlock-2/batch_normalization_5/moving_mean*
_output_shapes	
:*
dtype0
»
1ConvBlock-2/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31ConvBlock-2/batch_normalization_5/moving_variance
´
EConvBlock-2/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp1ConvBlock-2/batch_normalization_5/moving_variance*
_output_shapes	
:*
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

SGD/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameSGD/dense/kernel/momentum

-SGD/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/kernel/momentum* 
_output_shapes
:
*
dtype0

SGD/dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameSGD/dense/bias/momentum

+SGD/dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense/bias/momentum*
_output_shapes	
:*
dtype0

SGD/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_nameSGD/dense_1/kernel/momentum

/SGD/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/kernel/momentum* 
_output_shapes
:
*
dtype0

SGD/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_1/bias/momentum

-SGD/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/bias/momentum*
_output_shapes	
:*
dtype0

SGD/dense_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameSGD/dense_2/kernel/momentum

/SGD/dense_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/kernel/momentum*
_output_shapes
:	*
dtype0

SGD/dense_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_2/bias/momentum

-SGD/dense_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/bias/momentum*
_output_shapes
:*
dtype0
°
&SGD/ConvBlock-0/conv2d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&SGD/ConvBlock-0/conv2d/kernel/momentum
©
:SGD/ConvBlock-0/conv2d/kernel/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-0/conv2d/kernel/momentum*&
_output_shapes
: *
dtype0
 
$SGD/ConvBlock-0/conv2d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$SGD/ConvBlock-0/conv2d/bias/momentum

8SGD/ConvBlock-0/conv2d/bias/momentum/Read/ReadVariableOpReadVariableOp$SGD/ConvBlock-0/conv2d/bias/momentum*
_output_shapes
: *
dtype0
´
(SGD/ConvBlock-0/conv2d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *9
shared_name*(SGD/ConvBlock-0/conv2d_1/kernel/momentum
­
<SGD/ConvBlock-0/conv2d_1/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-0/conv2d_1/kernel/momentum*&
_output_shapes
:  *
dtype0
¤
&SGD/ConvBlock-0/conv2d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&SGD/ConvBlock-0/conv2d_1/bias/momentum

:SGD/ConvBlock-0/conv2d_1/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-0/conv2d_1/bias/momentum*
_output_shapes
: *
dtype0
¼
2SGD/ConvBlock-0/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42SGD/ConvBlock-0/batch_normalization/gamma/momentum
µ
FSGD/ConvBlock-0/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp2SGD/ConvBlock-0/batch_normalization/gamma/momentum*
_output_shapes
: *
dtype0
º
1SGD/ConvBlock-0/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31SGD/ConvBlock-0/batch_normalization/beta/momentum
³
ESGD/ConvBlock-0/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp1SGD/ConvBlock-0/batch_normalization/beta/momentum*
_output_shapes
: *
dtype0
À
4SGD/ConvBlock-0/batch_normalization_1/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64SGD/ConvBlock-0/batch_normalization_1/gamma/momentum
¹
HSGD/ConvBlock-0/batch_normalization_1/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-0/batch_normalization_1/gamma/momentum*
_output_shapes
: *
dtype0
¾
3SGD/ConvBlock-0/batch_normalization_1/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53SGD/ConvBlock-0/batch_normalization_1/beta/momentum
·
GSGD/ConvBlock-0/batch_normalization_1/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-0/batch_normalization_1/beta/momentum*
_output_shapes
: *
dtype0
´
(SGD/ConvBlock-1/conv2d_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:		 @*9
shared_name*(SGD/ConvBlock-1/conv2d_2/kernel/momentum
­
<SGD/ConvBlock-1/conv2d_2/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-1/conv2d_2/kernel/momentum*&
_output_shapes
:		 @*
dtype0
¤
&SGD/ConvBlock-1/conv2d_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&SGD/ConvBlock-1/conv2d_2/bias/momentum

:SGD/ConvBlock-1/conv2d_2/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-1/conv2d_2/bias/momentum*
_output_shapes
:@*
dtype0
´
(SGD/ConvBlock-1/conv2d_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@@*9
shared_name*(SGD/ConvBlock-1/conv2d_3/kernel/momentum
­
<SGD/ConvBlock-1/conv2d_3/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-1/conv2d_3/kernel/momentum*&
_output_shapes
:		@@*
dtype0
¤
&SGD/ConvBlock-1/conv2d_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&SGD/ConvBlock-1/conv2d_3/bias/momentum

:SGD/ConvBlock-1/conv2d_3/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-1/conv2d_3/bias/momentum*
_output_shapes
:@*
dtype0
À
4SGD/ConvBlock-1/batch_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64SGD/ConvBlock-1/batch_normalization_2/gamma/momentum
¹
HSGD/ConvBlock-1/batch_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-1/batch_normalization_2/gamma/momentum*
_output_shapes
:@*
dtype0
¾
3SGD/ConvBlock-1/batch_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53SGD/ConvBlock-1/batch_normalization_2/beta/momentum
·
GSGD/ConvBlock-1/batch_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-1/batch_normalization_2/beta/momentum*
_output_shapes
:@*
dtype0
À
4SGD/ConvBlock-1/batch_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64SGD/ConvBlock-1/batch_normalization_3/gamma/momentum
¹
HSGD/ConvBlock-1/batch_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-1/batch_normalization_3/gamma/momentum*
_output_shapes
:@*
dtype0
¾
3SGD/ConvBlock-1/batch_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53SGD/ConvBlock-1/batch_normalization_3/beta/momentum
·
GSGD/ConvBlock-1/batch_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-1/batch_normalization_3/beta/momentum*
_output_shapes
:@*
dtype0
µ
(SGD/ConvBlock-2/conv2d_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(SGD/ConvBlock-2/conv2d_4/kernel/momentum
®
<SGD/ConvBlock-2/conv2d_4/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-2/conv2d_4/kernel/momentum*'
_output_shapes
:@*
dtype0
¥
&SGD/ConvBlock-2/conv2d_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/ConvBlock-2/conv2d_4/bias/momentum

:SGD/ConvBlock-2/conv2d_4/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-2/conv2d_4/bias/momentum*
_output_shapes	
:*
dtype0
¶
(SGD/ConvBlock-2/conv2d_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(SGD/ConvBlock-2/conv2d_5/kernel/momentum
¯
<SGD/ConvBlock-2/conv2d_5/kernel/momentum/Read/ReadVariableOpReadVariableOp(SGD/ConvBlock-2/conv2d_5/kernel/momentum*(
_output_shapes
:*
dtype0
¥
&SGD/ConvBlock-2/conv2d_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&SGD/ConvBlock-2/conv2d_5/bias/momentum

:SGD/ConvBlock-2/conv2d_5/bias/momentum/Read/ReadVariableOpReadVariableOp&SGD/ConvBlock-2/conv2d_5/bias/momentum*
_output_shapes	
:*
dtype0
Á
4SGD/ConvBlock-2/batch_normalization_4/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64SGD/ConvBlock-2/batch_normalization_4/gamma/momentum
º
HSGD/ConvBlock-2/batch_normalization_4/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-2/batch_normalization_4/gamma/momentum*
_output_shapes	
:*
dtype0
¿
3SGD/ConvBlock-2/batch_normalization_4/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53SGD/ConvBlock-2/batch_normalization_4/beta/momentum
¸
GSGD/ConvBlock-2/batch_normalization_4/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-2/batch_normalization_4/beta/momentum*
_output_shapes	
:*
dtype0
Á
4SGD/ConvBlock-2/batch_normalization_5/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64SGD/ConvBlock-2/batch_normalization_5/gamma/momentum
º
HSGD/ConvBlock-2/batch_normalization_5/gamma/momentum/Read/ReadVariableOpReadVariableOp4SGD/ConvBlock-2/batch_normalization_5/gamma/momentum*
_output_shapes	
:*
dtype0
¿
3SGD/ConvBlock-2/batch_normalization_5/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53SGD/ConvBlock-2/batch_normalization_5/beta/momentum
¸
GSGD/ConvBlock-2/batch_normalization_5/beta/momentum/Read/ReadVariableOpReadVariableOp3SGD/ConvBlock-2/batch_normalization_5/beta/momentum*
_output_shapes	
:*
dtype0

NoOpNoOp
÷¸
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*±¸
value¦¸B¢¸ B¸
Õ
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
¶
	^decay
_learning_rate
`momentum
aiter<momentum§=momentum¨Jmomentum©KmomentumªXmomentum«Ymomentum¬bmomentum­cmomentum®dmomentum¯emomentum°fmomentum±gmomentum²hmomentum³imomentum´nmomentumµomomentum¶pmomentum·qmomentum¸rmomentum¹smomentumºtmomentum»umomentum¼zmomentum½{momentum¾|momentum¿}momentumÀ~momentumÁmomentumÂmomentumÃmomentumÄ
 
Ì
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
30
31
32
33
34
35
<36
=37
J38
K39
X40
Y41
è
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
22
23
<24
=25
J26
K27
X28
Y29
²
layers
regularization_losses
layer_metrics
	variables
 layer_regularization_losses
trainable_variables
non_trainable_variables
metrics
 


_state_var
 
 
 
²
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics

0
1

0
1

0
1
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
²
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
 trainable_variables
non_trainable_variables
metrics
 
 
 
²
layers
layer_metrics
"regularization_losses
 layer_regularization_losses
#	variables
$trainable_variables
non_trainable_variables
 metrics

¡0
¢1

£0
¤1

¥0
¦1
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
²
§layers
¨layer_metrics
)regularization_losses
 ©layer_regularization_losses
*	variables
+trainable_variables
ªnon_trainable_variables
«metrics
 
 
 
²
¬layers
­layer_metrics
-regularization_losses
 ®layer_regularization_losses
.	variables
/trainable_variables
¯non_trainable_variables
°metrics

±0
²1

³0
´1

µ0
¶1
 
\
z0
{1
|2
}3
~4
5
6
7
8
9
10
11
:
z0
{1
|2
}3
~4
5
6
7
²
·layers
¸layer_metrics
4regularization_losses
 ¹layer_regularization_losses
5	variables
6trainable_variables
ºnon_trainable_variables
»metrics
 
 
 
²
¼layers
½layer_metrics
8regularization_losses
 ¾layer_regularization_losses
9	variables
:trainable_variables
¿non_trainable_variables
Àmetrics
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
²
Álayers
Âlayer_metrics
>regularization_losses
 Ãlayer_regularization_losses
?	variables
@trainable_variables
Änon_trainable_variables
Åmetrics
 
 
 
²
Ælayers
Çlayer_metrics
Bregularization_losses
 Èlayer_regularization_losses
C	variables
Dtrainable_variables
Énon_trainable_variables
Êmetrics
 
 
 
²
Ëlayers
Ìlayer_metrics
Fregularization_losses
 Ílayer_regularization_losses
G	variables
Htrainable_variables
Înon_trainable_variables
Ïmetrics
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
²
Ðlayers
Ñlayer_metrics
Lregularization_losses
 Òlayer_regularization_losses
M	variables
Ntrainable_variables
Ónon_trainable_variables
Ômetrics
 
 
 
²
Õlayers
Ölayer_metrics
Pregularization_losses
 ×layer_regularization_losses
Q	variables
Rtrainable_variables
Ønon_trainable_variables
Ùmetrics
 
 
 
²
Úlayers
Ûlayer_metrics
Tregularization_losses
 Ülayer_regularization_losses
U	variables
Vtrainable_variables
Ýnon_trainable_variables
Þmetrics
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
²
ßlayers
àlayer_metrics
Zregularization_losses
 álayer_regularization_losses
[	variables
\trainable_variables
ânon_trainable_variables
ãmetrics
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
8
9
10
11
(
ä0
å1
æ2
ç3
è4
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
éregularization_losses
ê	variables
ëtrainable_variables
ì	keras_api
l

dkernel
ebias
íregularization_losses
î	variables
ïtrainable_variables
ð	keras_api
V
ñregularization_losses
ò	variables
ótrainable_variables
ô	keras_api
V
õregularization_losses
ö	variables
÷trainable_variables
ø	keras_api

	ùaxis
	fgamma
gbeta
jmoving_mean
kmoving_variance
úregularization_losses
û	variables
ütrainable_variables
ý	keras_api

	þaxis
	hgamma
ibeta
lmoving_mean
mmoving_variance
ÿregularization_losses
	variables
trainable_variables
	keras_api
0
0
1
2
3
4
5
 
 

j0
k1
l2
m3
 
 
 
 
 
 
l

nkernel
obias
regularization_losses
	variables
trainable_variables
	keras_api
l

pkernel
qbias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api

	axis
	rgamma
sbeta
vmoving_mean
wmoving_variance
regularization_losses
	variables
trainable_variables
	keras_api

	axis
	tgamma
ubeta
xmoving_mean
ymoving_variance
regularization_losses
	variables
trainable_variables
	keras_api
0
¡0
¢1
£2
¤3
¥4
¦5
 
 

v0
w1
x2
y3
 
 
 
 
 
 
l

zkernel
{bias
regularization_losses
	variables
trainable_variables
 	keras_api
l

|kernel
}bias
¡regularization_losses
¢	variables
£trainable_variables
¤	keras_api
V
¥regularization_losses
¦	variables
§trainable_variables
¨	keras_api
V
©regularization_losses
ª	variables
«trainable_variables
¬	keras_api

	­axis
	~gamma
beta
moving_mean
moving_variance
®regularization_losses
¯	variables
°trainable_variables
±	keras_api
 
	²axis

gamma
	beta
moving_mean
moving_variance
³regularization_losses
´	variables
µtrainable_variables
¶	keras_api
0
±0
²1
³2
´3
µ4
¶5
 
 
 
0
1
2
3
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

·total

¸count
¹	variables
º	keras_api
I

»total

¼count
½
_fn_kwargs
¾	variables
¿	keras_api
\
À
thresholds
Átrue_positives
Âfalse_positives
Ã	variables
Ä	keras_api
\
Å
thresholds
Ætrue_positives
Çfalse_negatives
È	variables
É	keras_api
/
Êtotal_cm
Ë	variables
Ì	keras_api
 

b0
c1

b0
c1
µ
Ílayers
Îlayer_metrics
éregularization_losses
 Ïlayer_regularization_losses
ê	variables
ëtrainable_variables
Ðnon_trainable_variables
Ñmetrics
 

d0
e1

d0
e1
µ
Òlayers
Ólayer_metrics
íregularization_losses
 Ôlayer_regularization_losses
î	variables
ïtrainable_variables
Õnon_trainable_variables
Ömetrics
 
 
 
µ
×layers
Ølayer_metrics
ñregularization_losses
 Ùlayer_regularization_losses
ò	variables
ótrainable_variables
Únon_trainable_variables
Ûmetrics
 
 
 
µ
Ülayers
Ýlayer_metrics
õregularization_losses
 Þlayer_regularization_losses
ö	variables
÷trainable_variables
ßnon_trainable_variables
àmetrics
 
 

f0
g1
j2
k3

f0
g1
µ
álayers
âlayer_metrics
úregularization_losses
 ãlayer_regularization_losses
û	variables
ütrainable_variables
änon_trainable_variables
åmetrics
 
 

h0
i1
l2
m3

h0
i1
µ
ælayers
çlayer_metrics
ÿregularization_losses
 èlayer_regularization_losses
	variables
trainable_variables
énon_trainable_variables
êmetrics
 

n0
o1

n0
o1
µ
ëlayers
ìlayer_metrics
regularization_losses
 ílayer_regularization_losses
	variables
trainable_variables
înon_trainable_variables
ïmetrics
 

p0
q1

p0
q1
µ
ðlayers
ñlayer_metrics
regularization_losses
 òlayer_regularization_losses
	variables
trainable_variables
ónon_trainable_variables
ômetrics
 
 
 
µ
õlayers
ölayer_metrics
regularization_losses
 ÷layer_regularization_losses
	variables
trainable_variables
ønon_trainable_variables
ùmetrics
 
 
 
µ
úlayers
ûlayer_metrics
regularization_losses
 ülayer_regularization_losses
	variables
trainable_variables
ýnon_trainable_variables
þmetrics
 
 

r0
s1
v2
w3

r0
s1
µ
ÿlayers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics
 
 

t0
u1
x2
y3

t0
u1
µ
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics
 

z0
{1

z0
{1
µ
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics
 

|0
}1

|0
}1
µ
layers
layer_metrics
¡regularization_losses
 layer_regularization_losses
¢	variables
£trainable_variables
non_trainable_variables
metrics
 
 
 
µ
layers
layer_metrics
¥regularization_losses
 layer_regularization_losses
¦	variables
§trainable_variables
non_trainable_variables
metrics
 
 
 
µ
layers
layer_metrics
©regularization_losses
 layer_regularization_losses
ª	variables
«trainable_variables
non_trainable_variables
metrics
 
 

~0
1
2
3

~0
1
µ
layers
layer_metrics
®regularization_losses
 layer_regularization_losses
¯	variables
°trainable_variables
 non_trainable_variables
¡metrics
 
 
 
0
1
2
3

0
1
µ
¢layers
£layer_metrics
³regularization_losses
 ¤layer_regularization_losses
´	variables
µtrainable_variables
¥non_trainable_variables
¦metrics
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

·0
¸1

¹	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

»0
¼1

¾	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

Á0
Â1

Ã	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

Æ0
Ç1

È	variables
US
VARIABLE_VALUEtotal_cm7keras_api/metrics/4/total_cm/.ATTRIBUTES/VARIABLE_VALUE

Ê0

Ë	variables
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
0
1
 
 
 
 

0
1
 

VARIABLE_VALUESGD/dense/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_1/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_1/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_2/kernel/momentumYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_2/bias/momentumWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&SGD/ConvBlock-0/conv2d/kernel/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$SGD/ConvBlock-0/conv2d/bias/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/ConvBlock-0/conv2d_1/kernel/momentumIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&SGD/ConvBlock-0/conv2d_1/bias/momentumIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2SGD/ConvBlock-0/batch_normalization/gamma/momentumIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE1SGD/ConvBlock-0/batch_normalization/beta/momentumIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE4SGD/ConvBlock-0/batch_normalization_1/gamma/momentumIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3SGD/ConvBlock-0/batch_normalization_1/beta/momentumIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/ConvBlock-1/conv2d_2/kernel/momentumJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&SGD/ConvBlock-1/conv2d_2/bias/momentumJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/ConvBlock-1/conv2d_3/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&SGD/ConvBlock-1/conv2d_3/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE4SGD/ConvBlock-1/batch_normalization_2/gamma/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3SGD/ConvBlock-1/batch_normalization_2/beta/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE4SGD/ConvBlock-1/batch_normalization_3/gamma/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3SGD/ConvBlock-1/batch_normalization_3/beta/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/ConvBlock-2/conv2d_4/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&SGD/ConvBlock-2/conv2d_4/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/ConvBlock-2/conv2d_5/kernel/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&SGD/ConvBlock-2/conv2d_5/bias/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE4SGD/ConvBlock-2/batch_normalization_4/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3SGD/ConvBlock-2/batch_normalization_4/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE4SGD/ConvBlock-2/batch_normalization_5/gamma/momentumJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3SGD/ConvBlock-2/batch_normalization_5/beta/momentumJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
×
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1ConvBlock-0/conv2d/kernelConvBlock-0/conv2d/bias%ConvBlock-0/batch_normalization/gamma$ConvBlock-0/batch_normalization/beta+ConvBlock-0/batch_normalization/moving_mean/ConvBlock-0/batch_normalization/moving_varianceConvBlock-0/conv2d_1/kernelConvBlock-0/conv2d_1/bias'ConvBlock-0/batch_normalization_1/gamma&ConvBlock-0/batch_normalization_1/beta-ConvBlock-0/batch_normalization_1/moving_mean1ConvBlock-0/batch_normalization_1/moving_varianceConvBlock-1/conv2d_2/kernelConvBlock-1/conv2d_2/bias'ConvBlock-1/batch_normalization_2/gamma&ConvBlock-1/batch_normalization_2/beta-ConvBlock-1/batch_normalization_2/moving_mean1ConvBlock-1/batch_normalization_2/moving_varianceConvBlock-1/conv2d_3/kernelConvBlock-1/conv2d_3/bias'ConvBlock-1/batch_normalization_3/gamma&ConvBlock-1/batch_normalization_3/beta-ConvBlock-1/batch_normalization_3/moving_mean1ConvBlock-1/batch_normalization_3/moving_varianceConvBlock-2/conv2d_4/kernelConvBlock-2/conv2d_4/bias'ConvBlock-2/batch_normalization_4/gamma&ConvBlock-2/batch_normalization_4/beta-ConvBlock-2/batch_normalization_4/moving_mean1ConvBlock-2/batch_normalization_4/moving_varianceConvBlock-2/conv2d_5/kernelConvBlock-2/conv2d_5/bias'ConvBlock-2/batch_normalization_5/gamma&ConvBlock-2/batch_normalization_5/beta-ConvBlock-2/batch_normalization_5/moving_mean1ConvBlock-2/batch_normalization_5/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_348044
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¦'
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
GPU2 *0J 8 *(
f#R!
__inference__traced_save_350610
é
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
GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_350878¡%
 
§
4__inference_batch_normalization_layer_call_fn_349961

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3455632
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
íð
6
"__inference__traced_restore_350878
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
identity_87¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_9Ü'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*è&
valueÞ&BÛ&WB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB2layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB7keras_api/metrics/4/total_cm/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¿
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*Ã
value¹B¶WB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesá
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ò
_output_shapesß
Ü:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*e
dtypes[
Y2W		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¦
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8 
AssignVariableOp_8AssignVariableOpassignvariableop_8_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_9 
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10µ
AssignVariableOp_10AssignVariableOp-assignvariableop_10_convblock_0_conv2d_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11³
AssignVariableOp_11AssignVariableOp+assignvariableop_11_convblock_0_conv2d_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_convblock_0_conv2d_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13µ
AssignVariableOp_13AssignVariableOp-assignvariableop_13_convblock_0_conv2d_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Á
AssignVariableOp_14AssignVariableOp9assignvariableop_14_convblock_0_batch_normalization_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15À
AssignVariableOp_15AssignVariableOp8assignvariableop_15_convblock_0_batch_normalization_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ã
AssignVariableOp_16AssignVariableOp;assignvariableop_16_convblock_0_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Â
AssignVariableOp_17AssignVariableOp:assignvariableop_17_convblock_0_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ç
AssignVariableOp_18AssignVariableOp?assignvariableop_18_convblock_0_batch_normalization_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ë
AssignVariableOp_19AssignVariableOpCassignvariableop_19_convblock_0_batch_normalization_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20É
AssignVariableOp_20AssignVariableOpAassignvariableop_20_convblock_0_batch_normalization_1_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Í
AssignVariableOp_21AssignVariableOpEassignvariableop_21_convblock_0_batch_normalization_1_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22·
AssignVariableOp_22AssignVariableOp/assignvariableop_22_convblock_1_conv2d_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23µ
AssignVariableOp_23AssignVariableOp-assignvariableop_23_convblock_1_conv2d_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24·
AssignVariableOp_24AssignVariableOp/assignvariableop_24_convblock_1_conv2d_3_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25µ
AssignVariableOp_25AssignVariableOp-assignvariableop_25_convblock_1_conv2d_3_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ã
AssignVariableOp_26AssignVariableOp;assignvariableop_26_convblock_1_batch_normalization_2_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Â
AssignVariableOp_27AssignVariableOp:assignvariableop_27_convblock_1_batch_normalization_2_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ã
AssignVariableOp_28AssignVariableOp;assignvariableop_28_convblock_1_batch_normalization_3_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Â
AssignVariableOp_29AssignVariableOp:assignvariableop_29_convblock_1_batch_normalization_3_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30É
AssignVariableOp_30AssignVariableOpAassignvariableop_30_convblock_1_batch_normalization_2_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Í
AssignVariableOp_31AssignVariableOpEassignvariableop_31_convblock_1_batch_normalization_2_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32É
AssignVariableOp_32AssignVariableOpAassignvariableop_32_convblock_1_batch_normalization_3_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Í
AssignVariableOp_33AssignVariableOpEassignvariableop_33_convblock_1_batch_normalization_3_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34·
AssignVariableOp_34AssignVariableOp/assignvariableop_34_convblock_2_conv2d_4_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35µ
AssignVariableOp_35AssignVariableOp-assignvariableop_35_convblock_2_conv2d_4_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36·
AssignVariableOp_36AssignVariableOp/assignvariableop_36_convblock_2_conv2d_5_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37µ
AssignVariableOp_37AssignVariableOp-assignvariableop_37_convblock_2_conv2d_5_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ã
AssignVariableOp_38AssignVariableOp;assignvariableop_38_convblock_2_batch_normalization_4_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Â
AssignVariableOp_39AssignVariableOp:assignvariableop_39_convblock_2_batch_normalization_4_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ã
AssignVariableOp_40AssignVariableOp;assignvariableop_40_convblock_2_batch_normalization_5_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Â
AssignVariableOp_41AssignVariableOp:assignvariableop_41_convblock_2_batch_normalization_5_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42É
AssignVariableOp_42AssignVariableOpAassignvariableop_42_convblock_2_batch_normalization_4_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Í
AssignVariableOp_43AssignVariableOpEassignvariableop_43_convblock_2_batch_normalization_4_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44É
AssignVariableOp_44AssignVariableOpAassignvariableop_44_convblock_2_batch_normalization_5_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Í
AssignVariableOp_45AssignVariableOpEassignvariableop_45_convblock_2_batch_normalization_5_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_46¤
AssignVariableOp_46AssignVariableOpassignvariableop_46_variableIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¡
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¡
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49£
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50£
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_1Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51ª
AssignVariableOp_51AssignVariableOp"assignvariableop_51_true_positivesIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52«
AssignVariableOp_52AssignVariableOp#assignvariableop_52_false_positivesIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¬
AssignVariableOp_53AssignVariableOp$assignvariableop_53_true_positives_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54«
AssignVariableOp_54AssignVariableOp#assignvariableop_54_false_negativesIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¤
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_cmIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56µ
AssignVariableOp_56AssignVariableOp-assignvariableop_56_sgd_dense_kernel_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57³
AssignVariableOp_57AssignVariableOp+assignvariableop_57_sgd_dense_bias_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58·
AssignVariableOp_58AssignVariableOp/assignvariableop_58_sgd_dense_1_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59µ
AssignVariableOp_59AssignVariableOp-assignvariableop_59_sgd_dense_1_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60·
AssignVariableOp_60AssignVariableOp/assignvariableop_60_sgd_dense_2_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61µ
AssignVariableOp_61AssignVariableOp-assignvariableop_61_sgd_dense_2_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Â
AssignVariableOp_62AssignVariableOp:assignvariableop_62_sgd_convblock_0_conv2d_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63À
AssignVariableOp_63AssignVariableOp8assignvariableop_63_sgd_convblock_0_conv2d_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ä
AssignVariableOp_64AssignVariableOp<assignvariableop_64_sgd_convblock_0_conv2d_1_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Â
AssignVariableOp_65AssignVariableOp:assignvariableop_65_sgd_convblock_0_conv2d_1_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Î
AssignVariableOp_66AssignVariableOpFassignvariableop_66_sgd_convblock_0_batch_normalization_gamma_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Í
AssignVariableOp_67AssignVariableOpEassignvariableop_67_sgd_convblock_0_batch_normalization_beta_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ð
AssignVariableOp_68AssignVariableOpHassignvariableop_68_sgd_convblock_0_batch_normalization_1_gamma_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ï
AssignVariableOp_69AssignVariableOpGassignvariableop_69_sgd_convblock_0_batch_normalization_1_beta_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ä
AssignVariableOp_70AssignVariableOp<assignvariableop_70_sgd_convblock_1_conv2d_2_kernel_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Â
AssignVariableOp_71AssignVariableOp:assignvariableop_71_sgd_convblock_1_conv2d_2_bias_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ä
AssignVariableOp_72AssignVariableOp<assignvariableop_72_sgd_convblock_1_conv2d_3_kernel_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Â
AssignVariableOp_73AssignVariableOp:assignvariableop_73_sgd_convblock_1_conv2d_3_bias_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Ð
AssignVariableOp_74AssignVariableOpHassignvariableop_74_sgd_convblock_1_batch_normalization_2_gamma_momentumIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Ï
AssignVariableOp_75AssignVariableOpGassignvariableop_75_sgd_convblock_1_batch_normalization_2_beta_momentumIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Ð
AssignVariableOp_76AssignVariableOpHassignvariableop_76_sgd_convblock_1_batch_normalization_3_gamma_momentumIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Ï
AssignVariableOp_77AssignVariableOpGassignvariableop_77_sgd_convblock_1_batch_normalization_3_beta_momentumIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Ä
AssignVariableOp_78AssignVariableOp<assignvariableop_78_sgd_convblock_2_conv2d_4_kernel_momentumIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Â
AssignVariableOp_79AssignVariableOp:assignvariableop_79_sgd_convblock_2_conv2d_4_bias_momentumIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Ä
AssignVariableOp_80AssignVariableOp<assignvariableop_80_sgd_convblock_2_conv2d_5_kernel_momentumIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81Â
AssignVariableOp_81AssignVariableOp:assignvariableop_81_sgd_convblock_2_conv2d_5_bias_momentumIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82Ð
AssignVariableOp_82AssignVariableOpHassignvariableop_82_sgd_convblock_2_batch_normalization_4_gamma_momentumIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83Ï
AssignVariableOp_83AssignVariableOpGassignvariableop_83_sgd_convblock_2_batch_normalization_4_beta_momentumIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84Ð
AssignVariableOp_84AssignVariableOpHassignvariableop_84_sgd_convblock_2_batch_normalization_5_gamma_momentumIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Ï
AssignVariableOp_85AssignVariableOpGassignvariableop_85_sgd_convblock_2_batch_normalization_5_beta_momentumIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_859
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÂ
Identity_86Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_86µ
Identity_87IdentityIdentity_86:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_87"#
identity_87Identity_87:output:0*ï
_input_shapesÝ
Ú: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
É
®
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_349981

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¹
¼
K__inference_random_rotation_layer_call_and_return_conditional_losses_346241

inputs-
)stateful_uniform_statefuluniform_resource
identity¢ stateful_uniform/StatefulUniformD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stateful_uniform/max
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape_dtype02"
 stateful_uniform/StatefulUniform
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¦
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6|
rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/zeros/mul/y¬
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
B :è2
rotation_matrix/zeros/Less/y§
rotation_matrix/zeros/LessLessrotation_matrix/zeros/mul:z:0%rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/Less
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
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
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceª
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2º
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©¿
ö

N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_347344
input_1
convblock_0_347174
convblock_0_347176
convblock_0_347178
convblock_0_347180
convblock_0_347182
convblock_0_347184
convblock_0_347186
convblock_0_347188
convblock_0_347190
convblock_0_347192
convblock_0_347194
convblock_0_347196
convblock_1_347200
convblock_1_347202
convblock_1_347204
convblock_1_347206
convblock_1_347208
convblock_1_347210
convblock_1_347212
convblock_1_347214
convblock_1_347216
convblock_1_347218
convblock_1_347220
convblock_1_347222
convblock_2_347226
convblock_2_347228
convblock_2_347230
convblock_2_347232
convblock_2_347234
convblock_2_347236
convblock_2_347238
convblock_2_347240
convblock_2_347242
convblock_2_347244
convblock_2_347246
convblock_2_347248
dense_347252
dense_347254
dense_1_347259
dense_1_347261
dense_2_347266
dense_2_347268
identity¢#ConvBlock-0/StatefulPartitionedCall¢#ConvBlock-1/StatefulPartitionedCall¢#ConvBlock-2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢+monte_carlo_dropout/StatefulPartitionedCall¢-monte_carlo_dropout_1/StatefulPartitionedCallù
random_rotation/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_3462452!
random_rotation/PartitionedCall®
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall(random_rotation/PartitionedCall:output:0convblock_0_347174convblock_0_347176convblock_0_347178convblock_0_347180convblock_0_347182convblock_0_347184convblock_0_347186convblock_0_347188convblock_0_347190convblock_0_347192convblock_0_347194convblock_0_347196*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_3464062%
#ConvBlock-0/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3456842
max_pooling2d/PartitionedCall¬
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_347200convblock_1_347202convblock_1_347204convblock_1_347206convblock_1_347208convblock_1_347210convblock_1_347212convblock_1_347214convblock_1_347216convblock_1_347218convblock_1_347220convblock_1_347222*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_3466362%
#ConvBlock-1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3459042!
max_pooling2d_1/PartitionedCall¯
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_347226convblock_2_347228convblock_2_347230convblock_2_347232convblock_2_347234convblock_2_347236convblock_2_347238convblock_2_347240convblock_2_347242convblock_2_347244convblock_2_347246convblock_2_347248*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_3468662%
#ConvBlock-2/StatefulPartitionedCall°
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_3461252*
(global_average_pooling2d/PartitionedCall¶
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_347252dense_347254*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3469622
dense/StatefulPartitionedCall³
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_3469902-
+monte_carlo_dropout/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_3470032
activation_6/PartitionedCall´
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_347259dense_1_347261*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3470212!
dense_1/StatefulPartitionedCallé
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_3470492/
-monte_carlo_dropout_1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_3470622
activation_7/PartitionedCall³
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_347266dense_2_347268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3470812!
dense_2/StatefulPartitionedCallÕ
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347174*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÅ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347176*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulÙ
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347186*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulÉ
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347188*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347200*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347202*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347212*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347214*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mulÚ
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347226*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347228*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulÛ
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347238*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347240*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul°
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ú
_input_shapesÈ
Å:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2J
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¤
©
6__inference_batch_normalization_3_layer_call_fn_350177

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3458872
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


O__inference_batch_normalization_layer_call_and_return_conditional_losses_345563

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñd

G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_346796
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
identity¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1¢$batch_normalization_5/AssignNewValue¢&batch_normalization_5/AssignNewValue_1±
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp»
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/BiasAdd
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_4/Relu·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ú
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_4/FusedBatchNormV3
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpä
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/BiasAdd
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_5/Relu·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ú
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_5/FusedBatchNormV3
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1ï
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulð
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul§
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ@::::::::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_1:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex
c

G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_346336
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
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1ª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
activation/Relu°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ç
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿvv : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3÷
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpá
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
conv2d_1/BiasAdd
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
activation_1/Relu¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1õ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿll : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_1/FusedBatchNormV3
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1è
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÙ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulî
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulß
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul¢
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_1:T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
£
I
-__inference_activation_7_layer_call_fn_349716

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_3470622
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_349999

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
½
m
4__inference_monte_carlo_dropout_layer_call_fn_349660

inputs
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_3469902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô­
Î,
__inference__traced_save_350610
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

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d5231e7cd1154fd78a681b919e2795dc/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÖ'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*è&
valueÞ&BÛ&WB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB2layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB7keras_api/metrics/4/total_cm/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/30/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/31/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¹
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:W*
dtype0*Ã
value¹B¶WB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop4savev2_convblock_0_conv2d_kernel_read_readvariableop2savev2_convblock_0_conv2d_bias_read_readvariableop6savev2_convblock_0_conv2d_1_kernel_read_readvariableop4savev2_convblock_0_conv2d_1_bias_read_readvariableop@savev2_convblock_0_batch_normalization_gamma_read_readvariableop?savev2_convblock_0_batch_normalization_beta_read_readvariableopBsavev2_convblock_0_batch_normalization_1_gamma_read_readvariableopAsavev2_convblock_0_batch_normalization_1_beta_read_readvariableopFsavev2_convblock_0_batch_normalization_moving_mean_read_readvariableopJsavev2_convblock_0_batch_normalization_moving_variance_read_readvariableopHsavev2_convblock_0_batch_normalization_1_moving_mean_read_readvariableopLsavev2_convblock_0_batch_normalization_1_moving_variance_read_readvariableop6savev2_convblock_1_conv2d_2_kernel_read_readvariableop4savev2_convblock_1_conv2d_2_bias_read_readvariableop6savev2_convblock_1_conv2d_3_kernel_read_readvariableop4savev2_convblock_1_conv2d_3_bias_read_readvariableopBsavev2_convblock_1_batch_normalization_2_gamma_read_readvariableopAsavev2_convblock_1_batch_normalization_2_beta_read_readvariableopBsavev2_convblock_1_batch_normalization_3_gamma_read_readvariableopAsavev2_convblock_1_batch_normalization_3_beta_read_readvariableopHsavev2_convblock_1_batch_normalization_2_moving_mean_read_readvariableopLsavev2_convblock_1_batch_normalization_2_moving_variance_read_readvariableopHsavev2_convblock_1_batch_normalization_3_moving_mean_read_readvariableopLsavev2_convblock_1_batch_normalization_3_moving_variance_read_readvariableop6savev2_convblock_2_conv2d_4_kernel_read_readvariableop4savev2_convblock_2_conv2d_4_bias_read_readvariableop6savev2_convblock_2_conv2d_5_kernel_read_readvariableop4savev2_convblock_2_conv2d_5_bias_read_readvariableopBsavev2_convblock_2_batch_normalization_4_gamma_read_readvariableopAsavev2_convblock_2_batch_normalization_4_beta_read_readvariableopBsavev2_convblock_2_batch_normalization_5_gamma_read_readvariableopAsavev2_convblock_2_batch_normalization_5_beta_read_readvariableopHsavev2_convblock_2_batch_normalization_4_moving_mean_read_readvariableopLsavev2_convblock_2_batch_normalization_4_moving_variance_read_readvariableopHsavev2_convblock_2_batch_normalization_5_moving_mean_read_readvariableopLsavev2_convblock_2_batch_normalization_5_moving_variance_read_readvariableop#savev2_variable_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop#savev2_total_cm_read_readvariableop4savev2_sgd_dense_kernel_momentum_read_readvariableop2savev2_sgd_dense_bias_momentum_read_readvariableop6savev2_sgd_dense_1_kernel_momentum_read_readvariableop4savev2_sgd_dense_1_bias_momentum_read_readvariableop6savev2_sgd_dense_2_kernel_momentum_read_readvariableop4savev2_sgd_dense_2_bias_momentum_read_readvariableopAsavev2_sgd_convblock_0_conv2d_kernel_momentum_read_readvariableop?savev2_sgd_convblock_0_conv2d_bias_momentum_read_readvariableopCsavev2_sgd_convblock_0_conv2d_1_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_0_conv2d_1_bias_momentum_read_readvariableopMsavev2_sgd_convblock_0_batch_normalization_gamma_momentum_read_readvariableopLsavev2_sgd_convblock_0_batch_normalization_beta_momentum_read_readvariableopOsavev2_sgd_convblock_0_batch_normalization_1_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_0_batch_normalization_1_beta_momentum_read_readvariableopCsavev2_sgd_convblock_1_conv2d_2_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_1_conv2d_2_bias_momentum_read_readvariableopCsavev2_sgd_convblock_1_conv2d_3_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_1_conv2d_3_bias_momentum_read_readvariableopOsavev2_sgd_convblock_1_batch_normalization_2_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_1_batch_normalization_2_beta_momentum_read_readvariableopOsavev2_sgd_convblock_1_batch_normalization_3_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_1_batch_normalization_3_beta_momentum_read_readvariableopCsavev2_sgd_convblock_2_conv2d_4_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_2_conv2d_4_bias_momentum_read_readvariableopCsavev2_sgd_convblock_2_conv2d_5_kernel_momentum_read_readvariableopAsavev2_sgd_convblock_2_conv2d_5_bias_momentum_read_readvariableopOsavev2_sgd_convblock_2_batch_normalization_4_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_2_batch_normalization_4_beta_momentum_read_readvariableopOsavev2_sgd_convblock_2_batch_normalization_5_gamma_momentum_read_readvariableopNsavev2_sgd_convblock_2_batch_normalization_5_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *e
dtypes[
Y2W		2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Í
_input_shapes»
¸: :
::
::	:: : : : : : :  : : : : : : : : : :		 @:@:		@@:@:@:@:@:@:@:@:@:@:@::::::::::::: : : : ::::::
::
::	:: : :  : : : : : :		 @:@:		@@:@:@:@:@:@:@:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 
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
:@:!$

_output_shapes	
::.%*
(
_output_shapes
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::!*

_output_shapes	
::!+

_output_shapes	
::!,

_output_shapes	
::!-

_output_shapes	
::!.

_output_shapes	
:: /
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
:!:

_output_shapes	
::&;"
 
_output_shapes
:
:!<

_output_shapes	
::%=!

_output_shapes
:	: >
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
:@:!P

_output_shapes	
::.Q*
(
_output_shapes
::!R

_output_shapes	
::!S

_output_shapes	
::!T

_output_shapes	
::!U

_output_shapes	
::!V

_output_shapes	
::W

_output_shapes
: 
¿	

,__inference_ConvBlock-0_layer_call_fn_349172
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_3464062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
à

3__inference_FERREIRA2020_class_layer_call_fn_348735

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
identity¢StatefulPartitionedCall«
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
:ÿÿÿÿÿÿÿÿÿ*@
_read_only_resource_inputs"
 	
 !"#&'()*+*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3475232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Þ
_input_shapesÌ
É:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
®
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_350133

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò
©
A__inference_dense_layer_call_and_return_conditional_losses_349634

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
«
C__inference_dense_2_layer_call_and_return_conditional_losses_349727

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
L
0__inference_random_rotation_layer_call_fn_348946

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_3462452
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
w
__inference_loss_fn_1_349763F
Bconvblock_0_conv2d_bias_regularizer_square_readvariableop_resource
identityõ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpBconvblock_0_conv2d_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
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
ç
|
__inference_loss_fn_10_349862J
Fconvblock_2_conv2d_5_kernel_regularizer_square_readvariableop_resource
identity
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_2_conv2d_5_kernel_regularizer_square_readvariableop_resource*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
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
²þ
Ø
!__inference__wrapped_model_345470
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
identity
;FERREIRA2020_class/ConvBlock-0/conv2d/Conv2D/ReadVariableOpReadVariableOpDferreira2020_class_convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;FERREIRA2020_class/ConvBlock-0/conv2d/Conv2D/ReadVariableOp
,FERREIRA2020_class/ConvBlock-0/conv2d/Conv2DConv2Dinput_1CFERREIRA2020_class/ConvBlock-0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv *
paddingVALID*
strides
2.
,FERREIRA2020_class/ConvBlock-0/conv2d/Conv2Dþ
<FERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd/ReadVariableOpReadVariableOpEferreira2020_class_convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<FERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd/ReadVariableOp 
-FERREIRA2020_class/ConvBlock-0/conv2d/BiasAddBiasAdd5FERREIRA2020_class/ConvBlock-0/conv2d/Conv2D:output:0DFERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2/
-FERREIRA2020_class/ConvBlock-0/conv2d/BiasAddÚ
.FERREIRA2020_class/ConvBlock-0/activation/ReluRelu6FERREIRA2020_class/ConvBlock-0/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 20
.FERREIRA2020_class/ConvBlock-0/activation/Relu
AFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOpReadVariableOpJferreira2020_class_convblock_0_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02C
AFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp
CFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp_1ReadVariableOpLferreira2020_class_convblock_0_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02E
CFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp_1À
RFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp[ferreira2020_class_convblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02T
RFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOpÆ
TFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]ferreira2020_class_convblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02V
TFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1²
CFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3FusedBatchNormV3<FERREIRA2020_class/ConvBlock-0/activation/Relu:activations:0IFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp:value:0KFERREIRA2020_class/ConvBlock-0/batch_normalization/ReadVariableOp_1:value:0ZFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0\FERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿvv : : : : :*
epsilon%o:*
is_training( 2E
CFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3
=FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D/ReadVariableOpÝ
.FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2DConv2DGFERREIRA2020_class/ConvBlock-0/batch_normalization/FusedBatchNormV3:y:0EFERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D
>FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp¨
/FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-0/conv2d_1/Conv2D:output:0FFERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 21
/FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAddà
0FERREIRA2020_class/ConvBlock-0/activation_1/ReluRelu8FERREIRA2020_class/ConvBlock-0/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 22
0FERREIRA2020_class/ConvBlock-0/activation_1/Relu
CFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOpReadVariableOpLferreira2020_class_convblock_0_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02E
CFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_0_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02G
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp_1Æ
TFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02V
TFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOpÌ
VFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02X
VFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1À
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-0/activation_1/Relu:activations:0KFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-0/batch_normalization_1/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿll : : : : :*
epsilon%o:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3
(FERREIRA2020_class/max_pooling2d/MaxPoolMaxPoolIFERREIRA2020_class/ConvBlock-0/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 *
ksize
*
paddingVALID*
strides
2*
(FERREIRA2020_class/max_pooling2d/MaxPool
=FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpÇ
.FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2DConv2D1FERREIRA2020_class/max_pooling2d/MaxPool:output:0EFERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D
>FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02@
>FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp¨
/FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-1/conv2d_2/Conv2D:output:0FFERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@21
/FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAddà
0FERREIRA2020_class/ConvBlock-1/activation_2/ReluRelu8FERREIRA2020_class/ConvBlock-1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@22
0FERREIRA2020_class/ConvBlock-1/activation_2/Relu
CFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOpReadVariableOpLferreira2020_class_convblock_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02E
CFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp_1Æ
TFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02V
TFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpÌ
VFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02X
VFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1À
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-1/activation_2/Relu:activations:0KFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-1/batch_normalization_2/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ..@:@:@:@:@:*
epsilon%o:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3
=FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D/ReadVariableOpß
.FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2DConv2DIFERREIRA2020_class/ConvBlock-1/batch_normalization_2/FusedBatchNormV3:y:0EFERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D
>FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02@
>FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp¨
/FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-1/conv2d_3/Conv2D:output:0FFERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@21
/FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAddà
0FERREIRA2020_class/ConvBlock-1/activation_3/ReluRelu8FERREIRA2020_class/ConvBlock-1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@22
0FERREIRA2020_class/ConvBlock-1/activation_3/Relu
CFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOpReadVariableOpLferreira2020_class_convblock_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02E
CFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp_1Æ
TFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02V
TFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpÌ
VFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02X
VFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1À
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-1/activation_3/Relu:activations:0KFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-1/batch_normalization_3/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ&&@:@:@:@:@:*
epsilon%o:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3
*FERREIRA2020_class/max_pooling2d_1/MaxPoolMaxPoolIFERREIRA2020_class/ConvBlock-1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2,
*FERREIRA2020_class/max_pooling2d_1/MaxPool
=FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpÊ
.FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2DConv2D3FERREIRA2020_class/max_pooling2d_1/MaxPool:output:0EFERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D
>FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02@
>FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp©
/FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-2/conv2d_4/Conv2D:output:0FFERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAddá
0FERREIRA2020_class/ConvBlock-2/activation_4/ReluRelu8FERREIRA2020_class/ConvBlock-2/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0FERREIRA2020_class/ConvBlock-2/activation_4/Relu
CFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOpReadVariableOpLferreira2020_class_convblock_2_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02E
CFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp_1Ç
TFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02V
TFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpÍ
VFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02X
VFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Å
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-2/activation_4/Relu:activations:0KFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-2/batch_normalization_4/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3
=FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D/ReadVariableOpReadVariableOpFferreira2020_class_convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02?
=FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D/ReadVariableOpà
.FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2DConv2DIFERREIRA2020_class/ConvBlock-2/batch_normalization_4/FusedBatchNormV3:y:0EFERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
20
.FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D
>FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpGferreira2020_class_convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02@
>FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp©
/FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAddBiasAdd7FERREIRA2020_class/ConvBlock-2/conv2d_5/Conv2D:output:0FFERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAddá
0FERREIRA2020_class/ConvBlock-2/activation_5/ReluRelu8FERREIRA2020_class/ConvBlock-2/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0FERREIRA2020_class/ConvBlock-2/activation_5/Relu
CFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOpReadVariableOpLferreira2020_class_convblock_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02E
CFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp_1ReadVariableOpNferreira2020_class_convblock_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp_1Ç
TFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp]ferreira2020_class_convblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02V
TFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpÍ
VFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp_ferreira2020_class_convblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02X
VFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Å
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3>FERREIRA2020_class/ConvBlock-2/activation_5/Relu:activations:0KFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp:value:0MFERREIRA2020_class/ConvBlock-2/batch_normalization_5/ReadVariableOp_1:value:0\FERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0^FERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2G
EFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3Ù
BFERREIRA2020_class/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2D
BFERREIRA2020_class/global_average_pooling2d/Mean/reduction_indices·
0FERREIRA2020_class/global_average_pooling2d/MeanMeanIFERREIRA2020_class/ConvBlock-2/batch_normalization_5/FusedBatchNormV3:y:0KFERREIRA2020_class/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0FERREIRA2020_class/global_average_pooling2d/MeanÚ
.FERREIRA2020_class/dense/MatMul/ReadVariableOpReadVariableOp7ferreira2020_class_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype020
.FERREIRA2020_class/dense/MatMul/ReadVariableOpò
FERREIRA2020_class/dense/MatMulMatMul9FERREIRA2020_class/global_average_pooling2d/Mean:output:06FERREIRA2020_class/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
FERREIRA2020_class/dense/MatMulØ
/FERREIRA2020_class/dense/BiasAdd/ReadVariableOpReadVariableOp8ferreira2020_class_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/FERREIRA2020_class/dense/BiasAdd/ReadVariableOpæ
 FERREIRA2020_class/dense/BiasAddBiasAdd)FERREIRA2020_class/dense/MatMul:product:07FERREIRA2020_class/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 FERREIRA2020_class/dense/BiasAdd±
4FERREIRA2020_class/monte_carlo_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @26
4FERREIRA2020_class/monte_carlo_dropout/dropout/Const
2FERREIRA2020_class/monte_carlo_dropout/dropout/MulMul)FERREIRA2020_class/dense/BiasAdd:output:0=FERREIRA2020_class/monte_carlo_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2FERREIRA2020_class/monte_carlo_dropout/dropout/MulÅ
4FERREIRA2020_class/monte_carlo_dropout/dropout/ShapeShape)FERREIRA2020_class/dense/BiasAdd:output:0*
T0*
_output_shapes
:26
4FERREIRA2020_class/monte_carlo_dropout/dropout/Shape·
KFERREIRA2020_class/monte_carlo_dropout/dropout/random_uniform/RandomUniformRandomUniform=FERREIRA2020_class/monte_carlo_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"2M
KFERREIRA2020_class/monte_carlo_dropout/dropout/random_uniform/RandomUniformÃ
=FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2?
=FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual/yÛ
;FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqualGreaterEqualTFERREIRA2020_class/monte_carlo_dropout/dropout/random_uniform/RandomUniform:output:0FFERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqualõ
3FERREIRA2020_class/monte_carlo_dropout/dropout/CastCast?FERREIRA2020_class/monte_carlo_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3FERREIRA2020_class/monte_carlo_dropout/dropout/Cast
4FERREIRA2020_class/monte_carlo_dropout/dropout/Mul_1Mul6FERREIRA2020_class/monte_carlo_dropout/dropout/Mul:z:07FERREIRA2020_class/monte_carlo_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4FERREIRA2020_class/monte_carlo_dropout/dropout/Mul_1Á
$FERREIRA2020_class/activation_6/ReluRelu8FERREIRA2020_class/monte_carlo_dropout/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$FERREIRA2020_class/activation_6/Reluà
0FERREIRA2020_class/dense_1/MatMul/ReadVariableOpReadVariableOp9ferreira2020_class_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype022
0FERREIRA2020_class/dense_1/MatMul/ReadVariableOpñ
!FERREIRA2020_class/dense_1/MatMulMatMul2FERREIRA2020_class/activation_6/Relu:activations:08FERREIRA2020_class/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!FERREIRA2020_class/dense_1/MatMulÞ
1FERREIRA2020_class/dense_1/BiasAdd/ReadVariableOpReadVariableOp:ferreira2020_class_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype023
1FERREIRA2020_class/dense_1/BiasAdd/ReadVariableOpî
"FERREIRA2020_class/dense_1/BiasAddBiasAdd+FERREIRA2020_class/dense_1/MatMul:product:09FERREIRA2020_class/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"FERREIRA2020_class/dense_1/BiasAddµ
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @28
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/Const
4FERREIRA2020_class/monte_carlo_dropout_1/dropout/MulMul+FERREIRA2020_class/dense_1/BiasAdd:output:0?FERREIRA2020_class/monte_carlo_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4FERREIRA2020_class/monte_carlo_dropout_1/dropout/MulË
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/ShapeShape+FERREIRA2020_class/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:28
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/ShapeÊ
MFERREIRA2020_class/monte_carlo_dropout_1/dropout/random_uniform/RandomUniformRandomUniform?FERREIRA2020_class/monte_carlo_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"*
seed22O
MFERREIRA2020_class/monte_carlo_dropout_1/dropout/random_uniform/RandomUniformÇ
?FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2A
?FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual/yã
=FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqualGreaterEqualVFERREIRA2020_class/monte_carlo_dropout_1/dropout/random_uniform/RandomUniform:output:0HFERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=FERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqualû
5FERREIRA2020_class/monte_carlo_dropout_1/dropout/CastCastAFERREIRA2020_class/monte_carlo_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5FERREIRA2020_class/monte_carlo_dropout_1/dropout/Cast
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul_1Mul8FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul:z:09FERREIRA2020_class/monte_carlo_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul_1Ã
$FERREIRA2020_class/activation_7/ReluRelu:FERREIRA2020_class/monte_carlo_dropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$FERREIRA2020_class/activation_7/Reluß
0FERREIRA2020_class/dense_2/MatMul/ReadVariableOpReadVariableOp9ferreira2020_class_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype022
0FERREIRA2020_class/dense_2/MatMul/ReadVariableOpð
!FERREIRA2020_class/dense_2/MatMulMatMul2FERREIRA2020_class/activation_7/Relu:activations:08FERREIRA2020_class/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!FERREIRA2020_class/dense_2/MatMulÝ
1FERREIRA2020_class/dense_2/BiasAdd/ReadVariableOpReadVariableOp:ferreira2020_class_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1FERREIRA2020_class/dense_2/BiasAdd/ReadVariableOpí
"FERREIRA2020_class/dense_2/BiasAddBiasAdd+FERREIRA2020_class/dense_2/MatMul:product:09FERREIRA2020_class/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"FERREIRA2020_class/dense_2/BiasAdd²
"FERREIRA2020_class/dense_2/SigmoidSigmoid+FERREIRA2020_class/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"FERREIRA2020_class/dense_2/Sigmoidz
IdentityIdentity&FERREIRA2020_class/dense_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ú
_input_shapesÈ
Å:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¡S
û
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_349566
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
identity±
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp»
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/BiasAdd
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_4/Relu·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ì
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpä
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/BiasAdd
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_5/Relu·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ì
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3ï
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulð
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ@:::::::::::::R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345904

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
g
K__inference_random_rotation_layer_call_and_return_conditional_losses_348934

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
p
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_347049

inputs
identityc
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
}
(__inference_dense_1_layer_call_fn_349689

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3470212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
{
&__inference_dense_layer_call_fn_349643

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3469622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
¼
K__inference_random_rotation_layer_call_and_return_conditional_losses_348930

inputs-
)stateful_uniform_statefuluniform_resource
identity¢ stateful_uniform/StatefulUniformD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stateful_uniform/max
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape_dtype02"
 stateful_uniform/StatefulUniform
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¦
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6|
rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/zeros/mul/y¬
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
B :è2
rotation_matrix/zeros/Less/y§
rotation_matrix/zeros/LessLessrotation_matrix/zeros/mul:z:0%rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/Less
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
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
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceª
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2º
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
I
-__inference_activation_6_layer_call_fn_349670

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_3470032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
{
__inference_loss_fn_6_349818J
Fconvblock_1_conv2d_3_kernel_regularizer_square_readvariableop_resource
identity
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_1_conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
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
Ç
¬
O__inference_batch_normalization_layer_call_and_return_conditional_losses_345532

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨
©
6__inference_batch_normalization_5_layer_call_fn_350329

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3461072
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àQ
ï
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_349114
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
identityª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
activation/Relu°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ù
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿvv : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpá
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
conv2d_1/BiasAdd
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
activation_1/Relu¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ç
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿll : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3è
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÙ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulî
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulß
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ:::::::::::::T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
Ô
«
C__inference_dense_1_layer_call_and_return_conditional_losses_347021

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»	

,__inference_ConvBlock-1_layer_call_fn_349398
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_3466362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ66 ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 

_user_specified_namex
¥

Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_346107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
{
__inference_loss_fn_8_349840J
Fconvblock_2_conv2d_4_kernel_regularizer_square_readvariableop_resource
identity
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_2_conv2d_4_kernel_regularizer_square_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
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
É
®
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345636

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
û
g
K__inference_random_rotation_layer_call_and_return_conditional_losses_346245

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
®
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_350069

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç
¬
O__inference_batch_normalization_layer_call_and_return_conditional_losses_349917

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º
d
H__inference_activation_7_layer_call_and_return_conditional_losses_349711

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
©
6__inference_batch_normalization_2_layer_call_fn_350100

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3457522
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_346076

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦¿
õ

N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_347788

inputs
convblock_0_347618
convblock_0_347620
convblock_0_347622
convblock_0_347624
convblock_0_347626
convblock_0_347628
convblock_0_347630
convblock_0_347632
convblock_0_347634
convblock_0_347636
convblock_0_347638
convblock_0_347640
convblock_1_347644
convblock_1_347646
convblock_1_347648
convblock_1_347650
convblock_1_347652
convblock_1_347654
convblock_1_347656
convblock_1_347658
convblock_1_347660
convblock_1_347662
convblock_1_347664
convblock_1_347666
convblock_2_347670
convblock_2_347672
convblock_2_347674
convblock_2_347676
convblock_2_347678
convblock_2_347680
convblock_2_347682
convblock_2_347684
convblock_2_347686
convblock_2_347688
convblock_2_347690
convblock_2_347692
dense_347696
dense_347698
dense_1_347703
dense_1_347705
dense_2_347710
dense_2_347712
identity¢#ConvBlock-0/StatefulPartitionedCall¢#ConvBlock-1/StatefulPartitionedCall¢#ConvBlock-2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢+monte_carlo_dropout/StatefulPartitionedCall¢-monte_carlo_dropout_1/StatefulPartitionedCallø
random_rotation/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_3462452!
random_rotation/PartitionedCall®
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall(random_rotation/PartitionedCall:output:0convblock_0_347618convblock_0_347620convblock_0_347622convblock_0_347624convblock_0_347626convblock_0_347628convblock_0_347630convblock_0_347632convblock_0_347634convblock_0_347636convblock_0_347638convblock_0_347640*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_3464062%
#ConvBlock-0/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3456842
max_pooling2d/PartitionedCall¬
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_347644convblock_1_347646convblock_1_347648convblock_1_347650convblock_1_347652convblock_1_347654convblock_1_347656convblock_1_347658convblock_1_347660convblock_1_347662convblock_1_347664convblock_1_347666*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_3466362%
#ConvBlock-1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3459042!
max_pooling2d_1/PartitionedCall¯
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_347670convblock_2_347672convblock_2_347674convblock_2_347676convblock_2_347678convblock_2_347680convblock_2_347682convblock_2_347684convblock_2_347686convblock_2_347688convblock_2_347690convblock_2_347692*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_3468662%
#ConvBlock-2/StatefulPartitionedCall°
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_3461252*
(global_average_pooling2d/PartitionedCall¶
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_347696dense_347698*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3469622
dense/StatefulPartitionedCall³
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_3469902-
+monte_carlo_dropout/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_3470032
activation_6/PartitionedCall´
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_347703dense_1_347705*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3470212!
dense_1/StatefulPartitionedCallé
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_3470492/
-monte_carlo_dropout_1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_3470622
activation_7/PartitionedCall³
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_347710dense_2_347712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3470812!
dense_2/StatefulPartitionedCallÕ
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347618*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÅ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347620*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulÙ
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347630*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulÉ
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347632*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347644*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347646*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347656*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347658*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mulÚ
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347670*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347672*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulÛ
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347682*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347684*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul°
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ú
_input_shapesÈ
Å:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::2J
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

3__inference_FERREIRA2020_class_layer_call_fn_347875
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
identity¢StatefulPartitionedCall«
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
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3477882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ú
_input_shapesÈ
Å:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Éd

G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_346566
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
identity¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpº
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_2/BiasAdd
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
activation_2/Relu¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1õ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ..@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_2/FusedBatchNormV3
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpã
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
conv2d_3/BiasAdd
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
activation_3/Relu¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1õ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ&&@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_3/FusedBatchNormV3
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1î
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulî
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul¦
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ66 ::::::::::::2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 

_user_specified_namex
É
®
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345752

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­
«
C__inference_dense_2_layer_call_and_return_conditional_losses_347081

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
{
__inference_loss_fn_4_349796J
Fconvblock_1_conv2d_2_kernel_regularizer_square_readvariableop_resource
identity
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_1_conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
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
¬¿
æ
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_348644

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
identityÎ
(ConvBlock-0/conv2d/Conv2D/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(ConvBlock-0/conv2d/Conv2D/ReadVariableOpÝ
ConvBlock-0/conv2d/Conv2DConv2Dinputs0ConvBlock-0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv *
paddingVALID*
strides
2
ConvBlock-0/conv2d/Conv2DÅ
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOpÔ
ConvBlock-0/conv2d/BiasAddBiasAdd"ConvBlock-0/conv2d/Conv2D:output:01ConvBlock-0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
ConvBlock-0/conv2d/BiasAdd¡
ConvBlock-0/activation/ReluRelu#ConvBlock-0/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
ConvBlock-0/activation/ReluÔ
.ConvBlock-0/batch_normalization/ReadVariableOpReadVariableOp7convblock_0_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype020
.ConvBlock-0/batch_normalization/ReadVariableOpÚ
0ConvBlock-0/batch_normalization/ReadVariableOp_1ReadVariableOp9convblock_0_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization/ReadVariableOp_1
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1­
0ConvBlock-0/batch_normalization/FusedBatchNormV3FusedBatchNormV3)ConvBlock-0/activation/Relu:activations:06ConvBlock-0/batch_normalization/ReadVariableOp:value:08ConvBlock-0/batch_normalization/ReadVariableOp_1:value:0GConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0IConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿvv : : : : :*
epsilon%o:*
is_training( 22
0ConvBlock-0/batch_normalization/FusedBatchNormV3Ô
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp
ConvBlock-0/conv2d_1/Conv2DConv2D4ConvBlock-0/batch_normalization/FusedBatchNormV3:y:02ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *
paddingVALID*
strides
2
ConvBlock-0/conv2d_1/Conv2DË
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpÜ
ConvBlock-0/conv2d_1/BiasAddBiasAdd$ConvBlock-0/conv2d_1/Conv2D:output:03ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
ConvBlock-0/conv2d_1/BiasAdd§
ConvBlock-0/activation_1/ReluRelu%ConvBlock-0/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
ConvBlock-0/activation_1/ReluÚ
0ConvBlock-0/batch_normalization_1/ReadVariableOpReadVariableOp9convblock_0_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization_1/ReadVariableOpà
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1ReadVariableOp;convblock_0_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1»
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3+ConvBlock-0/activation_1/Relu:activations:08ConvBlock-0/batch_normalization_1/ReadVariableOp:value:0:ConvBlock-0/batch_normalization_1/ReadVariableOp_1:value:0IConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿll : : : : :*
epsilon%o:*
is_training( 24
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3Þ
max_pooling2d/MaxPoolMaxPool6ConvBlock-0/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolÔ
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02,
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpû
ConvBlock-1/conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:02ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_2/Conv2DË
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpÜ
ConvBlock-1/conv2d_2/BiasAddBiasAdd$ConvBlock-1/conv2d_2/Conv2D:output:03ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
ConvBlock-1/conv2d_2/BiasAdd§
ConvBlock-1/activation_2/ReluRelu%ConvBlock-1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
ConvBlock-1/activation_2/ReluÚ
0ConvBlock-1/batch_normalization_2/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_2/ReadVariableOpà
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1»
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_2/Relu:activations:08ConvBlock-1/batch_normalization_2/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_2/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ..@:@:@:@:@:*
epsilon%o:*
is_training( 24
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3Ô
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02,
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp
ConvBlock-1/conv2d_3/Conv2DConv2D6ConvBlock-1/batch_normalization_2/FusedBatchNormV3:y:02ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_3/Conv2DË
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpÜ
ConvBlock-1/conv2d_3/BiasAddBiasAdd$ConvBlock-1/conv2d_3/Conv2D:output:03ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
ConvBlock-1/conv2d_3/BiasAdd§
ConvBlock-1/activation_3/ReluRelu%ConvBlock-1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
ConvBlock-1/activation_3/ReluÚ
0ConvBlock-1/batch_normalization_3/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_3/ReadVariableOpà
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1»
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_3/Relu:activations:08ConvBlock-1/batch_normalization_3/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_3/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ&&@:@:@:@:@:*
epsilon%o:*
is_training( 24
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3â
max_pooling2d_1/MaxPoolMaxPool6ConvBlock-1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolÕ
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpþ
ConvBlock-2/conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:02ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ConvBlock-2/conv2d_4/Conv2DÌ
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpÝ
ConvBlock-2/conv2d_4/BiasAddBiasAdd$ConvBlock-2/conv2d_4/Conv2D:output:03ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/conv2d_4/BiasAdd¨
ConvBlock-2/activation_4/ReluRelu%ConvBlock-2/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/activation_4/ReluÛ
0ConvBlock-2/batch_normalization_4/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype022
0ConvBlock-2/batch_normalization_4/ReadVariableOpá
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1À
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_4/Relu:activations:08ConvBlock-2/batch_normalization_4/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_4/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3Ö
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp
ConvBlock-2/conv2d_5/Conv2DConv2D6ConvBlock-2/batch_normalization_4/FusedBatchNormV3:y:02ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ConvBlock-2/conv2d_5/Conv2DÌ
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpÝ
ConvBlock-2/conv2d_5/BiasAddBiasAdd$ConvBlock-2/conv2d_5/Conv2D:output:03ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/conv2d_5/BiasAdd¨
ConvBlock-2/activation_5/ReluRelu%ConvBlock-2/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/activation_5/ReluÛ
0ConvBlock-2/batch_normalization_5/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype022
0ConvBlock-2/batch_normalization_5/ReadVariableOpá
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1À
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_5/Relu:activations:08ConvBlock-2/batch_normalization_5/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_5/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesë
global_average_pooling2d/MeanMean6ConvBlock-2/batch_normalization_5/FusedBatchNormV3:y:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_average_pooling2d/Mean¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp¦
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd
!monte_carlo_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2#
!monte_carlo_dropout/dropout/ConstÀ
monte_carlo_dropout/dropout/MulMuldense/BiasAdd:output:0*monte_carlo_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
monte_carlo_dropout/dropout/Mul
!monte_carlo_dropout/dropout/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2#
!monte_carlo_dropout/dropout/Shapeþ
8monte_carlo_dropout/dropout/random_uniform/RandomUniformRandomUniform*monte_carlo_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"2:
8monte_carlo_dropout/dropout/random_uniform/RandomUniform
*monte_carlo_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*monte_carlo_dropout/dropout/GreaterEqual/y
(monte_carlo_dropout/dropout/GreaterEqualGreaterEqualAmonte_carlo_dropout/dropout/random_uniform/RandomUniform:output:03monte_carlo_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(monte_carlo_dropout/dropout/GreaterEqual¼
 monte_carlo_dropout/dropout/CastCast,monte_carlo_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 monte_carlo_dropout/dropout/CastË
!monte_carlo_dropout/dropout/Mul_1Mul#monte_carlo_dropout/dropout/Mul:z:0$monte_carlo_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!monte_carlo_dropout/dropout/Mul_1
activation_6/ReluRelu%monte_carlo_dropout/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_6/Relu§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp¥
dense_1/MatMulMatMulactivation_6/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAdd
#monte_carlo_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#monte_carlo_dropout_1/dropout/ConstÈ
!monte_carlo_dropout_1/dropout/MulMuldense_1/BiasAdd:output:0,monte_carlo_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!monte_carlo_dropout_1/dropout/Mul
#monte_carlo_dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2%
#monte_carlo_dropout_1/dropout/Shape
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniformRandomUniform,monte_carlo_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"*
seed22<
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniform¡
,monte_carlo_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,monte_carlo_dropout_1/dropout/GreaterEqual/y
*monte_carlo_dropout_1/dropout/GreaterEqualGreaterEqualCmonte_carlo_dropout_1/dropout/random_uniform/RandomUniform:output:05monte_carlo_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*monte_carlo_dropout_1/dropout/GreaterEqualÂ
"monte_carlo_dropout_1/dropout/CastCast.monte_carlo_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"monte_carlo_dropout_1/dropout/CastÓ
#monte_carlo_dropout_1/dropout/Mul_1Mul%monte_carlo_dropout_1/dropout/Mul:z:0&monte_carlo_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#monte_carlo_dropout_1/dropout/Mul_1
activation_7/ReluRelu'monte_carlo_dropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_7/Relu¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp¤
dense_2/MatMulMatMulactivation_7/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Sigmoidô
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulå
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulú
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulë
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mulú
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulë
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulú
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulë
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mulû
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulì
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulü
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulì
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mulg
IdentityIdentitydense_2/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ú
_input_shapesÈ
Å:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345684

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
©
6__inference_batch_normalization_4_layer_call_fn_350265

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3460032
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
v
0__inference_random_rotation_layer_call_fn_348941

inputs
unknown
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_3462412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_345887

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Éd

G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_349270
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
identity¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpº
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_2/BiasAdd
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
activation_2/Relu¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1õ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ..@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_2/FusedBatchNormV3
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpã
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
conv2d_3/BiasAdd
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
activation_3/Relu¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1õ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ&&@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_3/FusedBatchNormV3
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1î
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulî
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul¦
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ66 ::::::::::::2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 

_user_specified_namex
¤
©
6__inference_batch_normalization_2_layer_call_fn_350113

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_3457832
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_345667

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¡
n
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_346990

inputs
identityc
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
y
__inference_loss_fn_0_349752H
Dconvblock_0_conv2d_kernel_regularizer_square_readvariableop_resource
identity
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_0_conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
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
ùR
û
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_346636
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
identity°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpº
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_2/BiasAdd
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
activation_2/Relu¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ç
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ..@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpã
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
conv2d_3/BiasAdd
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
activation_3/Relu¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ç
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ&&@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3î
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulî
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ66 :::::::::::::R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 

_user_specified_namex
ã

3__inference_FERREIRA2020_class_layer_call_fn_347612
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
identity¢StatefulPartitionedCall¬
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
:ÿÿÿÿÿÿÿÿÿ*@
_read_only_resource_inputs"
 	
 !"#&'()*+*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3475232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Þ
_input_shapesÌ
É:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
É
®
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_345856

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

y
__inference_loss_fn_3_349785H
Dconvblock_0_conv2d_1_bias_regularizer_square_readvariableop_resource
identityû
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_0_conv2d_1_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
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
¢
©
6__inference_batch_normalization_3_layer_call_fn_350164

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_3458562
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô
«
C__inference_dense_1_layer_call_and_return_conditional_losses_349680

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

U
9__inference_global_average_pooling2d_layer_call_fn_346131

inputs
identityà
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_3461252
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²­
Ä
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_348401

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
identity¢.ConvBlock-0/batch_normalization/AssignNewValue¢0ConvBlock-0/batch_normalization/AssignNewValue_1¢0ConvBlock-0/batch_normalization_1/AssignNewValue¢2ConvBlock-0/batch_normalization_1/AssignNewValue_1¢0ConvBlock-1/batch_normalization_2/AssignNewValue¢2ConvBlock-1/batch_normalization_2/AssignNewValue_1¢0ConvBlock-1/batch_normalization_3/AssignNewValue¢2ConvBlock-1/batch_normalization_3/AssignNewValue_1¢0ConvBlock-2/batch_normalization_4/AssignNewValue¢2ConvBlock-2/batch_normalization_4/AssignNewValue_1¢0ConvBlock-2/batch_normalization_5/AssignNewValue¢2ConvBlock-2/batch_normalization_5/AssignNewValue_1¢0random_rotation/stateful_uniform/StatefulUniformd
random_rotation/ShapeShapeinputs*
T0*
_output_shapes
:2
random_rotation/Shape
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2Â
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_1/stack
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_1
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2Ì
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_2/stack
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_1
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2Ì
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast_1®
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&random_rotation/stateful_uniform/shape
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2&
$random_rotation/stateful_uniform/min
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *:Fq?2&
$random_rotation/stateful_uniform/maxº
:random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:random_rotation/stateful_uniform/StatefulUniform/algorithmà
0random_rotation/stateful_uniform/StatefulUniformStatefulUniform9random_rotation_stateful_uniform_statefuluniform_resourceCrandom_rotation/stateful_uniform/StatefulUniform/algorithm:output:0/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape_dtype022
0random_rotation/stateful_uniform/StatefulUniformÒ
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/subæ
$random_rotation/stateful_uniform/mulMul9random_rotation/stateful_uniform/StatefulUniform:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$random_rotation/stateful_uniform/mulÒ
 random_rotation/stateful_uniformAdd(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_rotation/stateful_uniform
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%random_rotation/rotation_matrix/sub/y¾
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub¥
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Cos
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_1/yÄ
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1Ó
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/mul¥
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Sin
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_2/yÂ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2×
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_1×
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_3×
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_4
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/yê
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/rotation_matrix/truediv
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_5/yÂ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5©
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_1
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_6/yÄ
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6Ù
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_2©
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_1
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_7/yÂ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7Ù
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_3×
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/add×
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_8
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/yð
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation/rotation_matrix/truediv_1¢
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape´
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack¸
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1¸
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2¢
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice©
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_2¿
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stackÃ
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2×
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1©
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_2¿
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stackÃ
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2×
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2½
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Neg¿
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stackÃ
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2Ù
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3©
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_3¿
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stackÃ
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2×
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4©
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_3¿
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stackÃ
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2×
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5¿
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stackÃ
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2Û
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6
+random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/zeros/mul/yì
)random_rotation/rotation_matrix/zeros/mulMul6random_rotation/rotation_matrix/strided_slice:output:04random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2+
)random_rotation/rotation_matrix/zeros/mul
,random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2.
,random_rotation/rotation_matrix/zeros/Less/yç
*random_rotation/rotation_matrix/zeros/LessLess-random_rotation/rotation_matrix/zeros/mul:z:05random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2,
*random_rotation/rotation_matrix/zeros/Less¢
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Constõ
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/zeros
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axisÈ
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation/rotation_matrix/concatx
random_rotation/transform/ShapeShapeinputs*
T0*
_output_shapes
:2!
random_rotation/transform/Shape¨
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack¬
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1¬
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2ê
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_sliceê
4random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputs/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV2Î
(ConvBlock-0/conv2d/Conv2D/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(ConvBlock-0/conv2d/Conv2D/ReadVariableOp 
ConvBlock-0/conv2d/Conv2DConv2DIrandom_rotation/transform/ImageProjectiveTransformV2:transformed_images:00ConvBlock-0/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv *
paddingVALID*
strides
2
ConvBlock-0/conv2d/Conv2DÅ
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)ConvBlock-0/conv2d/BiasAdd/ReadVariableOpÔ
ConvBlock-0/conv2d/BiasAddBiasAdd"ConvBlock-0/conv2d/Conv2D:output:01ConvBlock-0/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
ConvBlock-0/conv2d/BiasAdd¡
ConvBlock-0/activation/ReluRelu#ConvBlock-0/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
ConvBlock-0/activation/ReluÔ
.ConvBlock-0/batch_normalization/ReadVariableOpReadVariableOp7convblock_0_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype020
.ConvBlock-0/batch_normalization/ReadVariableOpÚ
0ConvBlock-0/batch_normalization/ReadVariableOp_1ReadVariableOp9convblock_0_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization/ReadVariableOp_1
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1»
0ConvBlock-0/batch_normalization/FusedBatchNormV3FusedBatchNormV3)ConvBlock-0/activation/Relu:activations:06ConvBlock-0/batch_normalization/ReadVariableOp:value:08ConvBlock-0/batch_normalization/ReadVariableOp_1:value:0GConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0IConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿvv : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<22
0ConvBlock-0/batch_normalization/FusedBatchNormV3¿
.ConvBlock-0/batch_normalization/AssignNewValueAssignVariableOpHconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_resource=ConvBlock-0/batch_normalization/FusedBatchNormV3:batch_mean:0@^ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp*[
_classQ
OMloc:@ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype020
.ConvBlock-0/batch_normalization/AssignNewValueÍ
0ConvBlock-0/batch_normalization/AssignNewValue_1AssignVariableOpJconvblock_0_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceAConvBlock-0/batch_normalization/FusedBatchNormV3:batch_variance:0B^ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*]
_classS
QOloc:@ConvBlock-0/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype022
0ConvBlock-0/batch_normalization/AssignNewValue_1Ô
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02,
*ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp
ConvBlock-0/conv2d_1/Conv2DConv2D4ConvBlock-0/batch_normalization/FusedBatchNormV3:y:02ConvBlock-0/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *
paddingVALID*
strides
2
ConvBlock-0/conv2d_1/Conv2DË
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOpÜ
ConvBlock-0/conv2d_1/BiasAddBiasAdd$ConvBlock-0/conv2d_1/Conv2D:output:03ConvBlock-0/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
ConvBlock-0/conv2d_1/BiasAdd§
ConvBlock-0/activation_1/ReluRelu%ConvBlock-0/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
ConvBlock-0/activation_1/ReluÚ
0ConvBlock-0/batch_normalization_1/ReadVariableOpReadVariableOp9convblock_0_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype022
0ConvBlock-0/batch_normalization_1/ReadVariableOpà
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1ReadVariableOp;convblock_0_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype024
2ConvBlock-0/batch_normalization_1/ReadVariableOp_1
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
AConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
CConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1É
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3+ConvBlock-0/activation_1/Relu:activations:08ConvBlock-0/batch_normalization_1/ReadVariableOp:value:0:ConvBlock-0/batch_normalization_1/ReadVariableOp_1:value:0IConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿll : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<24
2ConvBlock-0/batch_normalization_1/FusedBatchNormV3Ë
0ConvBlock-0/batch_normalization_1/AssignNewValueAssignVariableOpJconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_resource?ConvBlock-0/batch_normalization_1/FusedBatchNormV3:batch_mean:0B^ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-0/batch_normalization_1/AssignNewValueÙ
2ConvBlock-0/batch_normalization_1/AssignNewValue_1AssignVariableOpLconvblock_0_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-0/batch_normalization_1/FusedBatchNormV3:batch_variance:0D^ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-0/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-0/batch_normalization_1/AssignNewValue_1Þ
max_pooling2d/MaxPoolMaxPool6ConvBlock-0/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolÔ
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02,
*ConvBlock-1/conv2d_2/Conv2D/ReadVariableOpû
ConvBlock-1/conv2d_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:02ConvBlock-1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_2/Conv2DË
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOpÜ
ConvBlock-1/conv2d_2/BiasAddBiasAdd$ConvBlock-1/conv2d_2/Conv2D:output:03ConvBlock-1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
ConvBlock-1/conv2d_2/BiasAdd§
ConvBlock-1/activation_2/ReluRelu%ConvBlock-1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
ConvBlock-1/activation_2/ReluÚ
0ConvBlock-1/batch_normalization_2/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_2/ReadVariableOpà
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_2/ReadVariableOp_1
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1É
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_2/Relu:activations:08ConvBlock-1/batch_normalization_2/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_2/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ..@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<24
2ConvBlock-1/batch_normalization_2/FusedBatchNormV3Ë
0ConvBlock-1/batch_normalization_2/AssignNewValueAssignVariableOpJconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource?ConvBlock-1/batch_normalization_2/FusedBatchNormV3:batch_mean:0B^ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-1/batch_normalization_2/AssignNewValueÙ
2ConvBlock-1/batch_normalization_2/AssignNewValue_1AssignVariableOpLconvblock_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-1/batch_normalization_2/FusedBatchNormV3:batch_variance:0D^ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-1/batch_normalization_2/AssignNewValue_1Ô
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02,
*ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp
ConvBlock-1/conv2d_3/Conv2DConv2D6ConvBlock-1/batch_normalization_2/FusedBatchNormV3:y:02ConvBlock-1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*
paddingVALID*
strides
2
ConvBlock-1/conv2d_3/Conv2DË
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOpÜ
ConvBlock-1/conv2d_3/BiasAddBiasAdd$ConvBlock-1/conv2d_3/Conv2D:output:03ConvBlock-1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
ConvBlock-1/conv2d_3/BiasAdd§
ConvBlock-1/activation_3/ReluRelu%ConvBlock-1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
ConvBlock-1/activation_3/ReluÚ
0ConvBlock-1/batch_normalization_3/ReadVariableOpReadVariableOp9convblock_1_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype022
0ConvBlock-1/batch_normalization_3/ReadVariableOpà
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1ReadVariableOp;convblock_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2ConvBlock-1/batch_normalization_3/ReadVariableOp_1
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02C
AConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02E
CConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1É
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+ConvBlock-1/activation_3/Relu:activations:08ConvBlock-1/batch_normalization_3/ReadVariableOp:value:0:ConvBlock-1/batch_normalization_3/ReadVariableOp_1:value:0IConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ&&@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<24
2ConvBlock-1/batch_normalization_3/FusedBatchNormV3Ë
0ConvBlock-1/batch_normalization_3/AssignNewValueAssignVariableOpJconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource?ConvBlock-1/batch_normalization_3/FusedBatchNormV3:batch_mean:0B^ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-1/batch_normalization_3/AssignNewValueÙ
2ConvBlock-1/batch_normalization_3/AssignNewValue_1AssignVariableOpLconvblock_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-1/batch_normalization_3/FusedBatchNormV3:batch_variance:0D^ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-1/batch_normalization_3/AssignNewValue_1â
max_pooling2d_1/MaxPoolMaxPool6ConvBlock-1/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolÕ
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*ConvBlock-2/conv2d_4/Conv2D/ReadVariableOpþ
ConvBlock-2/conv2d_4/Conv2DConv2D max_pooling2d_1/MaxPool:output:02ConvBlock-2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ConvBlock-2/conv2d_4/Conv2DÌ
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOpÝ
ConvBlock-2/conv2d_4/BiasAddBiasAdd$ConvBlock-2/conv2d_4/Conv2D:output:03ConvBlock-2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/conv2d_4/BiasAdd¨
ConvBlock-2/activation_4/ReluRelu%ConvBlock-2/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/activation_4/ReluÛ
0ConvBlock-2/batch_normalization_4/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype022
0ConvBlock-2/batch_normalization_4/ReadVariableOpá
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2ConvBlock-2/batch_normalization_4/ReadVariableOp_1
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
AConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
CConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Î
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_4/Relu:activations:08ConvBlock-2/batch_normalization_4/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_4/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<24
2ConvBlock-2/batch_normalization_4/FusedBatchNormV3Ë
0ConvBlock-2/batch_normalization_4/AssignNewValueAssignVariableOpJconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource?ConvBlock-2/batch_normalization_4/FusedBatchNormV3:batch_mean:0B^ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-2/batch_normalization_4/AssignNewValueÙ
2ConvBlock-2/batch_normalization_4/AssignNewValue_1AssignVariableOpLconvblock_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-2/batch_normalization_4/FusedBatchNormV3:batch_variance:0D^ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-2/batch_normalization_4/AssignNewValue_1Ö
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp
ConvBlock-2/conv2d_5/Conv2DConv2D6ConvBlock-2/batch_normalization_4/FusedBatchNormV3:y:02ConvBlock-2/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
ConvBlock-2/conv2d_5/Conv2DÌ
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOpÝ
ConvBlock-2/conv2d_5/BiasAddBiasAdd$ConvBlock-2/conv2d_5/Conv2D:output:03ConvBlock-2/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/conv2d_5/BiasAdd¨
ConvBlock-2/activation_5/ReluRelu%ConvBlock-2/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ConvBlock-2/activation_5/ReluÛ
0ConvBlock-2/batch_normalization_5/ReadVariableOpReadVariableOp9convblock_2_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype022
0ConvBlock-2/batch_normalization_5/ReadVariableOpá
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1ReadVariableOp;convblock_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2ConvBlock-2/batch_normalization_5/ReadVariableOp_1
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
AConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
CConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Î
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+ConvBlock-2/activation_5/Relu:activations:08ConvBlock-2/batch_normalization_5/ReadVariableOp:value:0:ConvBlock-2/batch_normalization_5/ReadVariableOp_1:value:0IConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0KConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<24
2ConvBlock-2/batch_normalization_5/FusedBatchNormV3Ë
0ConvBlock-2/batch_normalization_5/AssignNewValueAssignVariableOpJconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource?ConvBlock-2/batch_normalization_5/FusedBatchNormV3:batch_mean:0B^ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*]
_classS
QOloc:@ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype022
0ConvBlock-2/batch_normalization_5/AssignNewValueÙ
2ConvBlock-2/batch_normalization_5/AssignNewValue_1AssignVariableOpLconvblock_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceCConvBlock-2/batch_normalization_5/FusedBatchNormV3:batch_variance:0D^ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*_
_classU
SQloc:@ConvBlock-2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype024
2ConvBlock-2/batch_normalization_5/AssignNewValue_1³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesë
global_average_pooling2d/MeanMean6ConvBlock-2/batch_normalization_5/FusedBatchNormV3:y:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_average_pooling2d/Mean¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp¦
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd
!monte_carlo_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2#
!monte_carlo_dropout/dropout/ConstÀ
monte_carlo_dropout/dropout/MulMuldense/BiasAdd:output:0*monte_carlo_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
monte_carlo_dropout/dropout/Mul
!monte_carlo_dropout/dropout/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2#
!monte_carlo_dropout/dropout/Shapeþ
8monte_carlo_dropout/dropout/random_uniform/RandomUniformRandomUniform*monte_carlo_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"2:
8monte_carlo_dropout/dropout/random_uniform/RandomUniform
*monte_carlo_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*monte_carlo_dropout/dropout/GreaterEqual/y
(monte_carlo_dropout/dropout/GreaterEqualGreaterEqualAmonte_carlo_dropout/dropout/random_uniform/RandomUniform:output:03monte_carlo_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(monte_carlo_dropout/dropout/GreaterEqual¼
 monte_carlo_dropout/dropout/CastCast,monte_carlo_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 monte_carlo_dropout/dropout/CastË
!monte_carlo_dropout/dropout/Mul_1Mul#monte_carlo_dropout/dropout/Mul:z:0$monte_carlo_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!monte_carlo_dropout/dropout/Mul_1
activation_6/ReluRelu%monte_carlo_dropout/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_6/Relu§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp¥
dense_1/MatMulMatMulactivation_6/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAdd
#monte_carlo_dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#monte_carlo_dropout_1/dropout/ConstÈ
!monte_carlo_dropout_1/dropout/MulMuldense_1/BiasAdd:output:0,monte_carlo_dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!monte_carlo_dropout_1/dropout/Mul
#monte_carlo_dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2%
#monte_carlo_dropout_1/dropout/Shape
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniformRandomUniform,monte_carlo_dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"*
seed22<
:monte_carlo_dropout_1/dropout/random_uniform/RandomUniform¡
,monte_carlo_dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,monte_carlo_dropout_1/dropout/GreaterEqual/y
*monte_carlo_dropout_1/dropout/GreaterEqualGreaterEqualCmonte_carlo_dropout_1/dropout/random_uniform/RandomUniform:output:05monte_carlo_dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*monte_carlo_dropout_1/dropout/GreaterEqualÂ
"monte_carlo_dropout_1/dropout/CastCast.monte_carlo_dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"monte_carlo_dropout_1/dropout/CastÓ
#monte_carlo_dropout_1/dropout/Mul_1Mul%monte_carlo_dropout_1/dropout/Mul:z:0&monte_carlo_dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#monte_carlo_dropout_1/dropout/Mul_1
activation_7/ReluRelu'monte_carlo_dropout_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_7/Relu¦
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp¤
dense_2/MatMulMatMulactivation_7/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Sigmoidô
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1convblock_0_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulå
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp2convblock_0_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulú
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_0_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulë
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_0_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mulú
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulë
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulú
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulë
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mulû
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulì
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulü
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3convblock_2_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulì
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp4convblock_2_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul
IdentityIdentitydense_2/Sigmoid:y:0/^ConvBlock-0/batch_normalization/AssignNewValue1^ConvBlock-0/batch_normalization/AssignNewValue_11^ConvBlock-0/batch_normalization_1/AssignNewValue3^ConvBlock-0/batch_normalization_1/AssignNewValue_11^ConvBlock-1/batch_normalization_2/AssignNewValue3^ConvBlock-1/batch_normalization_2/AssignNewValue_11^ConvBlock-1/batch_normalization_3/AssignNewValue3^ConvBlock-1/batch_normalization_3/AssignNewValue_11^ConvBlock-2/batch_normalization_4/AssignNewValue3^ConvBlock-2/batch_normalization_4/AssignNewValue_11^ConvBlock-2/batch_normalization_5/AssignNewValue3^ConvBlock-2/batch_normalization_5/AssignNewValue_11^random_rotation/stateful_uniform/StatefulUniform*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Þ
_input_shapesÌ
É:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::2`
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

y
__inference_loss_fn_7_349829H
Dconvblock_1_conv2d_3_bias_regularizer_square_readvariableop_resource
identityû
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_1_conv2d_3_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
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
ùR
û
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_349340
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
identity°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpº
conv2d_2/Conv2DConv2Dx&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@*
paddingVALID*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
conv2d_2/BiasAdd
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ..@2
activation_2/Relu¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ç
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ..@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpã
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*
paddingVALID*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp¬
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
conv2d_3/BiasAdd
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2
activation_3/Relu¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ç
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ&&@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3î
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulî
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulß
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mul
IdentityIdentity*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ66 :::::::::::::R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 

_user_specified_namex
Á
¼
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_347170
input_1
random_rotation_346259
convblock_0_346466
convblock_0_346468
convblock_0_346470
convblock_0_346472
convblock_0_346474
convblock_0_346476
convblock_0_346478
convblock_0_346480
convblock_0_346482
convblock_0_346484
convblock_0_346486
convblock_0_346488
convblock_1_346696
convblock_1_346698
convblock_1_346700
convblock_1_346702
convblock_1_346704
convblock_1_346706
convblock_1_346708
convblock_1_346710
convblock_1_346712
convblock_1_346714
convblock_1_346716
convblock_1_346718
convblock_2_346926
convblock_2_346928
convblock_2_346930
convblock_2_346932
convblock_2_346934
convblock_2_346936
convblock_2_346938
convblock_2_346940
convblock_2_346942
convblock_2_346944
convblock_2_346946
convblock_2_346948
dense_346973
dense_346975
dense_1_347032
dense_1_347034
dense_2_347092
dense_2_347094
identity¢#ConvBlock-0/StatefulPartitionedCall¢#ConvBlock-1/StatefulPartitionedCall¢#ConvBlock-2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢+monte_carlo_dropout/StatefulPartitionedCall¢-monte_carlo_dropout_1/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCallª
'random_rotation/StatefulPartitionedCallStatefulPartitionedCallinput_1random_rotation_346259*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_3462412)
'random_rotation/StatefulPartitionedCall²
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0convblock_0_346466convblock_0_346468convblock_0_346470convblock_0_346472convblock_0_346474convblock_0_346476convblock_0_346478convblock_0_346480convblock_0_346482convblock_0_346484convblock_0_346486convblock_0_346488*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿll **
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_3463362%
#ConvBlock-0/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3456842
max_pooling2d/PartitionedCall¨
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_346696convblock_1_346698convblock_1_346700convblock_1_346702convblock_1_346704convblock_1_346706convblock_1_346708convblock_1_346710convblock_1_346712convblock_1_346714convblock_1_346716convblock_1_346718*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_3465662%
#ConvBlock-1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3459042!
max_pooling2d_1/PartitionedCall«
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_346926convblock_2_346928convblock_2_346930convblock_2_346932convblock_2_346934convblock_2_346936convblock_2_346938convblock_2_346940convblock_2_346942convblock_2_346944convblock_2_346946convblock_2_346948*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_3467962%
#ConvBlock-2/StatefulPartitionedCall°
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_3461252*
(global_average_pooling2d/PartitionedCall¶
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_346973dense_346975*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3469622
dense/StatefulPartitionedCall³
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_3469902-
+monte_carlo_dropout/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_3470032
activation_6/PartitionedCall´
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_347032dense_1_347034*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3470212!
dense_1/StatefulPartitionedCallé
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_3470492/
-monte_carlo_dropout_1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_3470622
activation_7/PartitionedCall³
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_347092dense_2_347094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3470812!
dense_2/StatefulPartitionedCallÕ
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_346466*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÅ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_346468*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulÙ
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_346478*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulÉ
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_346480*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_346696*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_346698*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_346708*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_346710*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mulÚ
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_346926*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_346928*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulÛ
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_346938*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_346940*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mulÚ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Þ
_input_shapesÌ
É:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::2J
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
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¦
©
6__inference_batch_normalization_4_layer_call_fn_350252

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_3459722
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_350285

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
d
H__inference_activation_6_layer_call_and_return_conditional_losses_347003

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
}
(__inference_dense_2_layer_call_fn_349736

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3470812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡S
û
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_346866
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
identity±
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp»
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/BiasAdd
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_4/Relu·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ì
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpä
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/BiasAdd
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_5/Relu·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ì
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3ï
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulð
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ@:::::::::::::R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex
»	

,__inference_ConvBlock-1_layer_call_fn_349369
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_3466362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ66 ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 

_user_specified_namex
¥

Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_350303

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_345783

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
£
p
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_349701

inputs
identityc
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñd

G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_349496
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
identity¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1¢$batch_normalization_5/AssignNewValue¢&batch_normalization_5/AssignNewValue_1±
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp»
conv2d_4/Conv2DConv2Dx&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/BiasAdd
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_4/Relu·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ú
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_4/FusedBatchNormV3
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpä
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/BiasAdd
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_5/Relu·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ú
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_5/FusedBatchNormV3
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1ï
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulð
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulà
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mul§
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ@::::::::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_1:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex


Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_350151

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½	

,__inference_ConvBlock-2_layer_call_fn_349624
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_3468662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex
º
d
H__inference_activation_6_layer_call_and_return_conditional_losses_349665

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½	

,__inference_ConvBlock-2_layer_call_fn_349595
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_3468662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ@::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex


Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_350087

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¥

Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_350239

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_346125

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_350221

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

y
__inference_loss_fn_9_349851H
Dconvblock_2_conv2d_4_bias_regularizer_square_readvariableop_resource
identityü
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_2_conv2d_4_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
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
®
J
.__inference_max_pooling2d_layer_call_fn_345690

inputs
identityï
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3456842
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
©
6__inference_batch_normalization_1_layer_call_fn_350012

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3456362
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤
©
6__inference_batch_normalization_1_layer_call_fn_350025

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_3456672
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
â
{
__inference_loss_fn_2_349774J
Fconvblock_0_conv2d_1_kernel_regularizer_square_readvariableop_resource
identity
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFconvblock_0_conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
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
²
L
0__inference_max_pooling2d_1_layer_call_fn_345910

inputs
identityñ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3459042
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àQ
ï
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_346406
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
identityª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
activation/Relu°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ù
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿvv : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpá
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
conv2d_1/BiasAdd
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
activation_1/Relu¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ç
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿll : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3è
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÙ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulî
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulß
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ:::::::::::::T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex

z
__inference_loss_fn_11_349873H
Dconvblock_2_conv2d_5_bias_regularizer_square_readvariableop_resource
identityü
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_2_conv2d_5_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
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
º
d
H__inference_activation_7_layer_call_and_return_conditional_losses_347062

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
©
A__inference_dense_layer_call_and_return_conditional_losses_346962

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿	

,__inference_ConvBlock-0_layer_call_fn_349143
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *.
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_3464062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex

y
__inference_loss_fn_5_349807H
Dconvblock_1_conv2d_2_bias_regularizer_square_readvariableop_resource
identityû
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpDconvblock_1_conv2d_2_bias_regularizer_square_readvariableop_resource*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
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
Á
o
6__inference_monte_carlo_dropout_1_layer_call_fn_349706

inputs
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_3470492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
ÿ
3__inference_FERREIRA2020_class_layer_call_fn_348824

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
identity¢StatefulPartitionedCallª
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
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3477882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ú
_input_shapesÈ
Å:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


O__inference_batch_normalization_layer_call_and_return_conditional_losses_349935

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
c

G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_349044
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
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1ª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
conv2d/BiasAdd}
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿvv 2
activation/Relu°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ç
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿvv : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3÷
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpá
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll *
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
conv2d_1/BiasAdd
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2
activation_1/Relu¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1õ
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿll : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_1/FusedBatchNormV3
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1è
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÙ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulî
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulß
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mul¢
IdentityIdentity*batch_normalization_1/FusedBatchNormV3:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿll 2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿ::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_1:T P
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namex
¥

Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_346003

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

§
4__inference_batch_normalization_layer_call_fn_349948

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3455322
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Õ
®
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_345972

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
n
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_349655

inputs
identityc
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seedò"2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
©
6__inference_batch_normalization_5_layer_call_fn_350316

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_3460762
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ñ
$__inference_signature_wrapper_348044
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
identity¢StatefulPartitionedCallþ
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
:ÿÿÿÿÿÿÿÿÿ*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_3454702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ú
_input_shapesÈ
Å:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Á
»
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_347523

inputs
random_rotation_347350
convblock_0_347353
convblock_0_347355
convblock_0_347357
convblock_0_347359
convblock_0_347361
convblock_0_347363
convblock_0_347365
convblock_0_347367
convblock_0_347369
convblock_0_347371
convblock_0_347373
convblock_0_347375
convblock_1_347379
convblock_1_347381
convblock_1_347383
convblock_1_347385
convblock_1_347387
convblock_1_347389
convblock_1_347391
convblock_1_347393
convblock_1_347395
convblock_1_347397
convblock_1_347399
convblock_1_347401
convblock_2_347405
convblock_2_347407
convblock_2_347409
convblock_2_347411
convblock_2_347413
convblock_2_347415
convblock_2_347417
convblock_2_347419
convblock_2_347421
convblock_2_347423
convblock_2_347425
convblock_2_347427
dense_347431
dense_347433
dense_1_347438
dense_1_347440
dense_2_347445
dense_2_347447
identity¢#ConvBlock-0/StatefulPartitionedCall¢#ConvBlock-1/StatefulPartitionedCall¢#ConvBlock-2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢+monte_carlo_dropout/StatefulPartitionedCall¢-monte_carlo_dropout_1/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCall©
'random_rotation/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_rotation_347350*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_3462412)
'random_rotation/StatefulPartitionedCall²
#ConvBlock-0/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0convblock_0_347353convblock_0_347355convblock_0_347357convblock_0_347359convblock_0_347361convblock_0_347363convblock_0_347365convblock_0_347367convblock_0_347369convblock_0_347371convblock_0_347373convblock_0_347375*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿll **
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_3463362%
#ConvBlock-0/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall,ConvBlock-0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ66 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3456842
max_pooling2d/PartitionedCall¨
#ConvBlock-1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0convblock_1_347379convblock_1_347381convblock_1_347383convblock_1_347385convblock_1_347387convblock_1_347389convblock_1_347391convblock_1_347393convblock_1_347395convblock_1_347397convblock_1_347399convblock_1_347401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&&@**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_3465662%
#ConvBlock-1/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall,ConvBlock-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3459042!
max_pooling2d_1/PartitionedCall«
#ConvBlock-2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0convblock_2_347405convblock_2_347407convblock_2_347409convblock_2_347411convblock_2_347413convblock_2_347415convblock_2_347417convblock_2_347419convblock_2_347421convblock_2_347423convblock_2_347425convblock_2_347427*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_3467962%
#ConvBlock-2/StatefulPartitionedCall°
(global_average_pooling2d/PartitionedCallPartitionedCall,ConvBlock-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_3461252*
(global_average_pooling2d/PartitionedCall¶
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_347431dense_347433*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3469622
dense/StatefulPartitionedCall³
+monte_carlo_dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_3469902-
+monte_carlo_dropout/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall4monte_carlo_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_3470032
activation_6/PartitionedCall´
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_1_347438dense_1_347440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3470212!
dense_1/StatefulPartitionedCallé
-monte_carlo_dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0,^monte_carlo_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_3470492/
-monte_carlo_dropout_1/StatefulPartitionedCall
activation_7/PartitionedCallPartitionedCall6monte_carlo_dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_3470622
activation_7/PartitionedCall³
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0dense_2_347445dense_2_347447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3470812!
dense_2/StatefulPartitionedCallÕ
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347353*&
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOpÜ
,ConvBlock-0/conv2d/kernel/Regularizer/SquareSquareCConvBlock-0/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,ConvBlock-0/conv2d/kernel/Regularizer/Square³
+ConvBlock-0/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+ConvBlock-0/conv2d/kernel/Regularizer/Constæ
)ConvBlock-0/conv2d/kernel/Regularizer/SumSum0ConvBlock-0/conv2d/kernel/Regularizer/Square:y:04ConvBlock-0/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/Sum
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d/kernel/Regularizer/mul/xè
)ConvBlock-0/conv2d/kernel/Regularizer/mulMul4ConvBlock-0/conv2d/kernel/Regularizer/mul/x:output:02ConvBlock-0/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d/kernel/Regularizer/mulÅ
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347355*
_output_shapes
: *
dtype02;
9ConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOpÊ
*ConvBlock-0/conv2d/bias/Regularizer/SquareSquareAConvBlock-0/conv2d/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2,
*ConvBlock-0/conv2d/bias/Regularizer/Square 
)ConvBlock-0/conv2d/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)ConvBlock-0/conv2d/bias/Regularizer/ConstÞ
'ConvBlock-0/conv2d/bias/Regularizer/SumSum.ConvBlock-0/conv2d/bias/Regularizer/Square:y:02ConvBlock-0/conv2d/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/Sum
)ConvBlock-0/conv2d/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2+
)ConvBlock-0/conv2d/bias/Regularizer/mul/xà
'ConvBlock-0/conv2d/bias/Regularizer/mulMul2ConvBlock-0/conv2d/bias/Regularizer/mul/x:output:00ConvBlock-0/conv2d/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'ConvBlock-0/conv2d/bias/Regularizer/mulÙ
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347365*&
_output_shapes
:  *
dtype02?
=ConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-0/conv2d_1/kernel/Regularizer/SquareSquareEConvBlock-0/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  20
.ConvBlock-0/conv2d_1/kernel/Regularizer/Square·
-ConvBlock-0/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/Constî
+ConvBlock-0/conv2d_1/kernel/Regularizer/SumSum2ConvBlock-0/conv2d_1/kernel/Regularizer/Square:y:06ConvBlock-0/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/Sum£
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-0/conv2d_1/kernel/Regularizer/mul/xð
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulMul6ConvBlock-0/conv2d_1/kernel/Regularizer/mul/x:output:04ConvBlock-0/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-0/conv2d_1/kernel/Regularizer/mulÉ
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_0_347367*
_output_shapes
: *
dtype02=
;ConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-0/conv2d_1/bias/Regularizer/SquareSquareCConvBlock-0/conv2d_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,ConvBlock-0/conv2d_1/bias/Regularizer/Square¤
+ConvBlock-0/conv2d_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-0/conv2d_1/bias/Regularizer/Constæ
)ConvBlock-0/conv2d_1/bias/Regularizer/SumSum0ConvBlock-0/conv2d_1/bias/Regularizer/Square:y:04ConvBlock-0/conv2d_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/Sum
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-0/conv2d_1/bias/Regularizer/mul/xè
)ConvBlock-0/conv2d_1/bias/Regularizer/mulMul4ConvBlock-0/conv2d_1/bias/Regularizer/mul/x:output:02ConvBlock-0/conv2d_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-0/conv2d_1/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347379*&
_output_shapes
:		 @*
dtype02?
=ConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_2/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		 @20
.ConvBlock-1/conv2d_2/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_2/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_2/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_2/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_2/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_2/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347381*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_2/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_2/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_2/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_2/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_2/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_2/bias/Regularizer/SumSum0ConvBlock-1/conv2d_2/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_2/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/Sum
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_2/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_2/bias/Regularizer/mulMul4ConvBlock-1/conv2d_2/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_2/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_2/bias/Regularizer/mulÙ
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347391*&
_output_shapes
:		@@*
dtype02?
=ConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOpâ
.ConvBlock-1/conv2d_3/kernel/Regularizer/SquareSquareEConvBlock-1/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:		@@20
.ConvBlock-1/conv2d_3/kernel/Regularizer/Square·
-ConvBlock-1/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/Constî
+ConvBlock-1/conv2d_3/kernel/Regularizer/SumSum2ConvBlock-1/conv2d_3/kernel/Regularizer/Square:y:06ConvBlock-1/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/Sum£
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-1/conv2d_3/kernel/Regularizer/mul/xð
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulMul6ConvBlock-1/conv2d_3/kernel/Regularizer/mul/x:output:04ConvBlock-1/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-1/conv2d_3/kernel/Regularizer/mulÉ
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_1_347393*
_output_shapes
:@*
dtype02=
;ConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOpÐ
,ConvBlock-1/conv2d_3/bias/Regularizer/SquareSquareCConvBlock-1/conv2d_3/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:@2.
,ConvBlock-1/conv2d_3/bias/Regularizer/Square¤
+ConvBlock-1/conv2d_3/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-1/conv2d_3/bias/Regularizer/Constæ
)ConvBlock-1/conv2d_3/bias/Regularizer/SumSum0ConvBlock-1/conv2d_3/bias/Regularizer/Square:y:04ConvBlock-1/conv2d_3/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/Sum
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-1/conv2d_3/bias/Regularizer/mul/xè
)ConvBlock-1/conv2d_3/bias/Regularizer/mulMul4ConvBlock-1/conv2d_3/bias/Regularizer/mul/x:output:02ConvBlock-1/conv2d_3/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-1/conv2d_3/bias/Regularizer/mulÚ
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347405*'
_output_shapes
:@*
dtype02?
=ConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOpã
.ConvBlock-2/conv2d_4/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:@20
.ConvBlock-2/conv2d_4/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_4/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_4/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_4/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_4/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_4/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347407*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_4/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_4/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_4/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_4/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_4/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_4/bias/Regularizer/SumSum0ConvBlock-2/conv2d_4/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_4/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/Sum
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_4/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_4/bias/Regularizer/mulMul4ConvBlock-2/conv2d_4/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_4/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_4/bias/Regularizer/mulÛ
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347417*(
_output_shapes
:*
dtype02?
=ConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOpä
.ConvBlock-2/conv2d_5/kernel/Regularizer/SquareSquareEConvBlock-2/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*(
_output_shapes
:20
.ConvBlock-2/conv2d_5/kernel/Regularizer/Square·
-ConvBlock-2/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/Constî
+ConvBlock-2/conv2d_5/kernel/Regularizer/SumSum2ConvBlock-2/conv2d_5/kernel/Regularizer/Square:y:06ConvBlock-2/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/Sum£
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2/
-ConvBlock-2/conv2d_5/kernel/Regularizer/mul/xð
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulMul6ConvBlock-2/conv2d_5/kernel/Regularizer/mul/x:output:04ConvBlock-2/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+ConvBlock-2/conv2d_5/kernel/Regularizer/mulÊ
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpReadVariableOpconvblock_2_347419*
_output_shapes	
:*
dtype02=
;ConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOpÑ
,ConvBlock-2/conv2d_5/bias/Regularizer/SquareSquareCConvBlock-2/conv2d_5/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2.
,ConvBlock-2/conv2d_5/bias/Regularizer/Square¤
+ConvBlock-2/conv2d_5/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+ConvBlock-2/conv2d_5/bias/Regularizer/Constæ
)ConvBlock-2/conv2d_5/bias/Regularizer/SumSum0ConvBlock-2/conv2d_5/bias/Regularizer/Square:y:04ConvBlock-2/conv2d_5/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/Sum
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2-
+ConvBlock-2/conv2d_5/bias/Regularizer/mul/xè
)ConvBlock-2/conv2d_5/bias/Regularizer/mulMul4ConvBlock-2/conv2d_5/bias/Regularizer/mul/x:output:02ConvBlock-2/conv2d_5/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)ConvBlock-2/conv2d_5/bias/Regularizer/mulÚ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0$^ConvBlock-0/StatefulPartitionedCall$^ConvBlock-1/StatefulPartitionedCall$^ConvBlock-2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^monte_carlo_dropout/StatefulPartitionedCall.^monte_carlo_dropout_1/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Þ
_input_shapesÌ
É:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::::::2J
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Åõ
ê0
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
+Å&call_and_return_all_conditional_losses
Æ__call__
Ç_default_save_signature"¸,
_tf_keras_network,{"class_name": "Functional", "name": "FERREIRA2020_class", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "FERREIRA2020_class", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": {"class_name": "__tuple__", "items": [0.15, 0.15]}, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}, "name": "random_rotation", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "ConvBlock", "config": {"layer was saved without config": true}, "name": "ConvBlock-0", "inbound_nodes": [[["random_rotation", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["ConvBlock-0", 0, 0, {}]]]}, {"class_name": "ConvBlock", "config": {"layer was saved without config": true}, "name": "ConvBlock-1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["ConvBlock-1", 0, 0, {}]]]}, {"class_name": "ConvBlock", "config": {"layer was saved without config": true}, "name": "ConvBlock-2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["ConvBlock-2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "MonteCarloDropout", "config": {"name": "monte_carlo_dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "monte_carlo_dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["monte_carlo_dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "MonteCarloDropout", "config": {"name": "monte_carlo_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "monte_carlo_dropout_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["monte_carlo_dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": ["accuracy", {"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Upsilon", "config": {"name": "upsilon", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 1.570841959619429e-05, "decay": 9.999999747378752e-05, "momentum": 0.8999999761581421, "nesterov": false}}}}
ý"ú
_tf_keras_input_layerÚ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Õ
_rng
regularization_losses
	variables
trainable_variables
	keras_api
+È&call_and_return_all_conditional_losses
É__call__"º
_tf_keras_layer {"class_name": "RandomRotation", "name": "random_rotation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": {"class_name": "__tuple__", "items": [0.15, 0.15]}, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}
Ô

conv2d

activation

batch_norm
regularization_losses
	variables
 trainable_variables
!	keras_api
+Ê&call_and_return_all_conditional_losses
Ë__call__"
_tf_keras_layerý{"class_name": "ConvBlock", "name": "ConvBlock-0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ý
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+Ì&call_and_return_all_conditional_losses
Í__call__"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ô

&conv2d
'
activation
(
batch_norm
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+Î&call_and_return_all_conditional_losses
Ï__call__"
_tf_keras_layerý{"class_name": "ConvBlock", "name": "ConvBlock-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

-regularization_losses
.	variables
/trainable_variables
0	keras_api
+Ð&call_and_return_all_conditional_losses
Ñ__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ô

1conv2d
2
activation
3
batch_norm
4regularization_losses
5	variables
6trainable_variables
7	keras_api
+Ò&call_and_return_all_conditional_losses
Ó__call__"
_tf_keras_layerý{"class_name": "ConvBlock", "name": "ConvBlock-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

8regularization_losses
9	variables
:trainable_variables
;	keras_api
+Ô&call_and_return_all_conditional_losses
Õ__call__"
_tf_keras_layerê{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ó

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+Ö&call_and_return_all_conditional_losses
×__call__"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}

Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+Ø&call_and_return_all_conditional_losses
Ù__call__"õ
_tf_keras_layerÛ{"class_name": "MonteCarloDropout", "name": "monte_carlo_dropout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "monte_carlo_dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
×
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+Ú&call_and_return_all_conditional_losses
Û__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}
÷

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+Ü&call_and_return_all_conditional_losses
Ý__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}

Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+Þ&call_and_return_all_conditional_losses
ß__call__"ù
_tf_keras_layerß{"class_name": "MonteCarloDropout", "name": "monte_carlo_dropout_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "monte_carlo_dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
×
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+à&call_and_return_all_conditional_losses
á__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}}
ö

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+â&call_and_return_all_conditional_losses
ã__call__"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
É
	^decay
_learning_rate
`momentum
aiter<momentum§=momentum¨Jmomentum©KmomentumªXmomentum«Ymomentum¬bmomentum­cmomentum®dmomentum¯emomentum°fmomentum±gmomentum²hmomentum³imomentum´nmomentumµomomentum¶pmomentum·qmomentum¸rmomentum¹smomentumºtmomentum»umomentum¼zmomentum½{momentum¾|momentum¿}momentumÀ~momentumÁmomentumÂmomentumÃmomentumÄ"
	optimizer
 "
trackable_list_wrapper
ì
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
30
31
32
33
34
35
<36
=37
J38
K39
X40
Y41"
trackable_list_wrapper

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
22
23
<24
=25
J26
K27
X28
Y29"
trackable_list_wrapper
Ó
layers
regularization_losses
layer_metrics
	variables
 layer_regularization_losses
trainable_variables
non_trainable_variables
metrics
Æ__call__
Ç_default_save_signature
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
-
äserving_default"
signature_map
/

_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics
É__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
@
å0
æ1
ç2
è3"
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
µ
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
 trainable_variables
non_trainable_variables
metrics
Ë__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
layer_metrics
"regularization_losses
 layer_regularization_losses
#	variables
$trainable_variables
non_trainable_variables
 metrics
Í__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
0
¡0
¢1"
trackable_list_wrapper
0
£0
¤1"
trackable_list_wrapper
0
¥0
¦1"
trackable_list_wrapper
@
é0
ê1
ë2
ì3"
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
µ
§layers
¨layer_metrics
)regularization_losses
 ©layer_regularization_losses
*	variables
+trainable_variables
ªnon_trainable_variables
«metrics
Ï__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¬layers
­layer_metrics
-regularization_losses
 ®layer_regularization_losses
.	variables
/trainable_variables
¯non_trainable_variables
°metrics
Ñ__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
0
±0
²1"
trackable_list_wrapper
0
³0
´1"
trackable_list_wrapper
0
µ0
¶1"
trackable_list_wrapper
@
í0
î1
ï2
ð3"
trackable_list_wrapper
|
z0
{1
|2
}3
~4
5
6
7
8
9
10
11"
trackable_list_wrapper
Z
z0
{1
|2
}3
~4
5
6
7"
trackable_list_wrapper
µ
·layers
¸layer_metrics
4regularization_losses
 ¹layer_regularization_losses
5	variables
6trainable_variables
ºnon_trainable_variables
»metrics
Ó__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¼layers
½layer_metrics
8regularization_losses
 ¾layer_regularization_losses
9	variables
:trainable_variables
¿non_trainable_variables
Àmetrics
Õ__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
 :
2dense/kernel
:2
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
µ
Álayers
Âlayer_metrics
>regularization_losses
 Ãlayer_regularization_losses
?	variables
@trainable_variables
Änon_trainable_variables
Åmetrics
×__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ælayers
Çlayer_metrics
Bregularization_losses
 Èlayer_regularization_losses
C	variables
Dtrainable_variables
Énon_trainable_variables
Êmetrics
Ù__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ëlayers
Ìlayer_metrics
Fregularization_losses
 Ílayer_regularization_losses
G	variables
Htrainable_variables
Înon_trainable_variables
Ïmetrics
Û__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
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
µ
Ðlayers
Ñlayer_metrics
Lregularization_losses
 Òlayer_regularization_losses
M	variables
Ntrainable_variables
Ónon_trainable_variables
Ômetrics
Ý__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Õlayers
Ölayer_metrics
Pregularization_losses
 ×layer_regularization_losses
Q	variables
Rtrainable_variables
Ønon_trainable_variables
Ùmetrics
ß__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Úlayers
Ûlayer_metrics
Tregularization_losses
 Ülayer_regularization_losses
U	variables
Vtrainable_variables
Ýnon_trainable_variables
Þmetrics
á__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
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
µ
ßlayers
àlayer_metrics
Zregularization_losses
 álayer_regularization_losses
[	variables
\trainable_variables
ânon_trainable_variables
ãmetrics
ã__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
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
6:4@2ConvBlock-2/conv2d_4/kernel
(:&2ConvBlock-2/conv2d_4/bias
7:52ConvBlock-2/conv2d_5/kernel
(:&2ConvBlock-2/conv2d_5/bias
6:42'ConvBlock-2/batch_normalization_4/gamma
5:32&ConvBlock-2/batch_normalization_4/beta
6:42'ConvBlock-2/batch_normalization_5/gamma
5:32&ConvBlock-2/batch_normalization_5/beta
>:< (2-ConvBlock-2/batch_normalization_4/moving_mean
B:@ (21ConvBlock-2/batch_normalization_4/moving_variance
>:< (2-ConvBlock-2/batch_normalization_5/moving_mean
B:@ (21ConvBlock-2/batch_normalization_5/moving_variance

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
trackable_dict_wrapper
 "
trackable_list_wrapper
z
j0
k1
l2
m3
v4
w5
x6
y7
8
9
10
11"
trackable_list_wrapper
H
ä0
å1
æ2
ç3
è4"
trackable_list_wrapper
:	2Variable
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
ç


bkernel
cbias
éregularization_losses
ê	variables
ëtrainable_variables
ì	keras_api
+ñ&call_and_return_all_conditional_losses
ò__call__"¼	
_tf_keras_layer¢	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 1]}}
í


dkernel
ebias
íregularization_losses
î	variables
ïtrainable_variables
ð	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"Â	
_tf_keras_layer¨	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 118, 118, 32]}}
×
ñregularization_losses
ò	variables
ótrainable_variables
ô	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"Â
_tf_keras_layer¨{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
Û
õregularization_losses
ö	variables
÷trainable_variables
ø	keras_api
+÷&call_and_return_all_conditional_losses
ø__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
¿	
	ùaxis
	fgamma
gbeta
jmoving_mean
kmoving_variance
úregularization_losses
û	variables
ütrainable_variables
ý	keras_api
+ù&call_and_return_all_conditional_losses
ú__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 118, 118, 32]}}
Ã	
	þaxis
	hgamma
ibeta
lmoving_mean
mmoving_variance
ÿregularization_losses
	variables
trainable_variables
	keras_api
+û&call_and_return_all_conditional_losses
ü__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 108, 108, 32]}}
P
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
j0
k1
l2
m3"
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
é


nkernel
obias
regularization_losses
	variables
trainable_variables
	keras_api
+ý&call_and_return_all_conditional_losses
þ__call__"¾	
_tf_keras_layer¤	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 54, 54, 32]}}
é


pkernel
qbias
regularization_losses
	variables
trainable_variables
	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"¾	
_tf_keras_layer¤	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 64]}}
Û
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
Û
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
Á	
	axis
	rgamma
sbeta
vmoving_mean
wmoving_variance
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 64]}}
Á	
	axis
	tgamma
ubeta
xmoving_mean
ymoving_variance
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 38, 38, 64]}}
P
¡0
¢1
£2
¤3
¥4
¦5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
v0
w1
x2
y3"
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
ê


zkernel
{bias
regularization_losses
	variables
trainable_variables
 	keras_api
+&call_and_return_all_conditional_losses
__call__"¿	
_tf_keras_layer¥	{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 64]}}
ì


|kernel
}bias
¡regularization_losses
¢	variables
£trainable_variables
¤	keras_api
+&call_and_return_all_conditional_losses
__call__"Á	
_tf_keras_layer§	{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.10000000149011612}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 128]}}
Û
¥regularization_losses
¦	variables
§trainable_variables
¨	keras_api
+&call_and_return_all_conditional_losses
__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
Û
©regularization_losses
ª	variables
«trainable_variables
¬	keras_api
+&call_and_return_all_conditional_losses
__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
Å	
	­axis
	~gamma
beta
moving_mean
moving_variance
®regularization_losses
¯	variables
°trainable_variables
±	keras_api
+&call_and_return_all_conditional_losses
__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 128]}}
Å	
	²axis

gamma
	beta
moving_mean
moving_variance
³regularization_losses
´	variables
µtrainable_variables
¶	keras_api
+&call_and_return_all_conditional_losses
__call__"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 128]}}
P
±0
²1
³2
´3
µ4
¶5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

·total

¸count
¹	variables
º	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


»total

¼count
½
_fn_kwargs
¾	variables
¿	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
¨
À
thresholds
Átrue_positives
Âfalse_positives
Ã	variables
Ä	keras_api"É
_tf_keras_metric®{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}

Å
thresholds
Ætrue_positives
Çfalse_negatives
È	variables
É	keras_api"À
_tf_keras_metric¥{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
¿
Êtotal_cm
Ë	variables
Ì	keras_api"
_tf_keras_metrics{"class_name": "Upsilon", "name": "upsilon", "dtype": "float32", "config": {"name": "upsilon", "dtype": "float32"}}
0
å0
æ1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
¸
Ílayers
Îlayer_metrics
éregularization_losses
 Ïlayer_regularization_losses
ê	variables
ëtrainable_variables
Ðnon_trainable_variables
Ñmetrics
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
0
ç0
è1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
¸
Òlayers
Ólayer_metrics
íregularization_losses
 Ôlayer_regularization_losses
î	variables
ïtrainable_variables
Õnon_trainable_variables
Ömetrics
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
×layers
Ølayer_metrics
ñregularization_losses
 Ùlayer_regularization_losses
ò	variables
ótrainable_variables
Únon_trainable_variables
Ûmetrics
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ülayers
Ýlayer_metrics
õregularization_losses
 Þlayer_regularization_losses
ö	variables
÷trainable_variables
ßnon_trainable_variables
àmetrics
ø__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
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
¸
álayers
âlayer_metrics
úregularization_losses
 ãlayer_regularization_losses
û	variables
ütrainable_variables
änon_trainable_variables
åmetrics
ú__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
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
¸
ælayers
çlayer_metrics
ÿregularization_losses
 èlayer_regularization_losses
	variables
trainable_variables
énon_trainable_variables
êmetrics
ü__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
0
é0
ê1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
¸
ëlayers
ìlayer_metrics
regularization_losses
 ílayer_regularization_losses
	variables
trainable_variables
înon_trainable_variables
ïmetrics
þ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
0
ë0
ì1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
¸
ðlayers
ñlayer_metrics
regularization_losses
 òlayer_regularization_losses
	variables
trainable_variables
ónon_trainable_variables
ômetrics
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
õlayers
ölayer_metrics
regularization_losses
 ÷layer_regularization_losses
	variables
trainable_variables
ønon_trainable_variables
ùmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
úlayers
ûlayer_metrics
regularization_losses
 ülayer_regularization_losses
	variables
trainable_variables
ýnon_trainable_variables
þmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
¸
ÿlayers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
¸
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0
í0
î1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
¸
layers
layer_metrics
regularization_losses
 layer_regularization_losses
	variables
trainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
0
ï0
ð1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
¸
layers
layer_metrics
¡regularization_losses
 layer_regularization_losses
¢	variables
£trainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
¥regularization_losses
 layer_regularization_losses
¦	variables
§trainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
layer_metrics
©regularization_losses
 layer_regularization_losses
ª	variables
«trainable_variables
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
>
~0
1
2
3"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
¸
layers
layer_metrics
®regularization_losses
 layer_regularization_losses
¯	variables
°trainable_variables
 non_trainable_variables
¡metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¢layers
£layer_metrics
³regularization_losses
 ¤layer_regularization_losses
´	variables
µtrainable_variables
¥non_trainable_variables
¦metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
0
·0
¸1"
trackable_list_wrapper
.
¹	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
»0
¼1"
trackable_list_wrapper
.
¾	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Á0
Â1"
trackable_list_wrapper
.
Ã	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Æ0
Ç1"
trackable_list_wrapper
.
È	variables"
_generic_user_object
: (2total_cm
(
Ê0"
trackable_list_wrapper
.
Ë	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
å0
æ1"
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
ç0
è1"
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
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
é0
ê1"
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
ë0
ì1"
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
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
í0
î1"
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
ï0
ð1"
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
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
+:)
2SGD/dense/kernel/momentum
$:"2SGD/dense/bias/momentum
-:+
2SGD/dense_1/kernel/momentum
&:$2SGD/dense_1/bias/momentum
,:*	2SGD/dense_2/kernel/momentum
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
A:?@2(SGD/ConvBlock-2/conv2d_4/kernel/momentum
3:12&SGD/ConvBlock-2/conv2d_4/bias/momentum
B:@2(SGD/ConvBlock-2/conv2d_5/kernel/momentum
3:12&SGD/ConvBlock-2/conv2d_5/bias/momentum
A:?24SGD/ConvBlock-2/batch_normalization_4/gamma/momentum
@:>23SGD/ConvBlock-2/batch_normalization_4/beta/momentum
A:?24SGD/ConvBlock-2/batch_normalization_5/gamma/momentum
@:>23SGD/ConvBlock-2/batch_normalization_5/beta/momentum
2
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_348401
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_348644
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_347170
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_347344À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
3__inference_FERREIRA2020_class_layer_call_fn_348824
3__inference_FERREIRA2020_class_layer_call_fn_348735
3__inference_FERREIRA2020_class_layer_call_fn_347875
3__inference_FERREIRA2020_class_layer_call_fn_347612À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
é2æ
!__inference__wrapped_model_345470À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ
Ô2Ñ
K__inference_random_rotation_layer_call_and_return_conditional_losses_348934
K__inference_random_rotation_layer_call_and_return_conditional_losses_348930´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_random_rotation_layer_call_fn_348946
0__inference_random_rotation_layer_call_fn_348941´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_349044
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_349114®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_ConvBlock-0_layer_call_fn_349143
,__inference_ConvBlock-0_layer_call_fn_349172®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±2®
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345684à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_max_pooling2d_layer_call_fn_345690à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Æ2Ã
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_349270
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_349340®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_ConvBlock-1_layer_call_fn_349398
,__inference_ConvBlock-1_layer_call_fn_349369®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345904à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
0__inference_max_pooling2d_1_layer_call_fn_345910à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Æ2Ã
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_349496
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_349566®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_ConvBlock-2_layer_call_fn_349624
,__inference_ConvBlock-2_layer_call_fn_349595®
¥²¡
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¼2¹
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_346125à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¡2
9__inference_global_average_pooling2d_layer_call_fn_346131à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_349634¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_349643¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ù2ö
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_349655¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
4__inference_monte_carlo_dropout_layer_call_fn_349660¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_6_layer_call_and_return_conditional_losses_349665¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_6_layer_call_fn_349670¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_349680¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_349689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_349701¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à2Ý
6__inference_monte_carlo_dropout_1_layer_call_fn_349706¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_7_layer_call_and_return_conditional_losses_349711¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_7_layer_call_fn_349716¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_349727¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_349736¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
3B1
$__inference_signature_wrapper_348044input_1
³2°
__inference_loss_fn_0_349752
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_349763
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_349774
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_3_349785
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_4_349796
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_5_349807
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_6_349818
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_7_349829
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_8_349840
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_9_349851
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_10_349862
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_11_349873
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
O__inference_batch_normalization_layer_call_and_return_conditional_losses_349917
O__inference_batch_normalization_layer_call_and_return_conditional_losses_349935´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
4__inference_batch_normalization_layer_call_fn_349948
4__inference_batch_normalization_layer_call_fn_349961´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_349981
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_349999´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_1_layer_call_fn_350012
6__inference_batch_normalization_1_layer_call_fn_350025´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_350087
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_350069´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_2_layer_call_fn_350100
6__inference_batch_normalization_2_layer_call_fn_350113´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_350151
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_350133´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_3_layer_call_fn_350177
6__inference_batch_normalization_3_layer_call_fn_350164´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_350239
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_350221´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_4_layer_call_fn_350252
6__inference_batch_normalization_4_layer_call_fn_350265´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_350285
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_350303´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ª2§
6__inference_batch_normalization_5_layer_call_fn_350329
6__inference_batch_normalization_5_layer_call_fn_350316´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 Â
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_349044wbcfgjkdehilm8¢5
.¢+
%"
xÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿll 
 Â
G__inference_ConvBlock-0_layer_call_and_return_conditional_losses_349114wbcfgjkdehilm8¢5
.¢+
%"
xÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿll 
 
,__inference_ConvBlock-0_layer_call_fn_349143jbcfgjkdehilm8¢5
.¢+
%"
xÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿll 
,__inference_ConvBlock-0_layer_call_fn_349172jbcfgjkdehilm8¢5
.¢+
%"
xÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿll À
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_349270unorsvwpqtuxy6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ66 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ&&@
 À
G__inference_ConvBlock-1_layer_call_and_return_conditional_losses_349340unorsvwpqtuxy6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ66 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ&&@
 
,__inference_ConvBlock-1_layer_call_fn_349369hnorsvwpqtuxy6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ66 
p
ª " ÿÿÿÿÿÿÿÿÿ&&@
,__inference_ConvBlock-1_layer_call_fn_349398hnorsvwpqtuxy6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ66 
p 
ª " ÿÿÿÿÿÿÿÿÿ&&@Ç
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_349496|z{~|}6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ@
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ç
G__inference_ConvBlock-2_layer_call_and_return_conditional_losses_349566|z{~|}6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ@
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_ConvBlock-2_layer_call_fn_349595oz{~|}6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ@
p
ª "!ÿÿÿÿÿÿÿÿÿ
,__inference_ConvBlock-2_layer_call_fn_349624oz{~|}6¢3
,¢)
# 
xÿÿÿÿÿÿÿÿÿ@
p 
ª "!ÿÿÿÿÿÿÿÿÿò
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3471702bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ð
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3473440bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ñ
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3484012bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ï
N__inference_FERREIRA2020_class_layer_call_and_return_conditional_losses_3486440bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
3__inference_FERREIRA2020_class_layer_call_fn_3476122bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÈ
3__inference_FERREIRA2020_class_layer_call_fn_3478750bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÉ
3__inference_FERREIRA2020_class_layer_call_fn_3487352bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÇ
3__inference_FERREIRA2020_class_layer_call_fn_3488240bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÇ
!__inference__wrapped_model_345470¡0bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXY:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ¦
H__inference_activation_6_layer_call_and_return_conditional_losses_349665Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
-__inference_activation_6_layer_call_fn_349670M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
H__inference_activation_7_layer_call_and_return_conditional_losses_349711Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
-__inference_activation_7_layer_call_fn_349716M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿì
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_349981hilmM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ì
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_349999hilmM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ä
6__inference_batch_normalization_1_layer_call_fn_350012hilmM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ä
6__inference_batch_normalization_1_layer_call_fn_350025hilmM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ì
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_350069rsvwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ì
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_350087rsvwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ä
6__inference_batch_normalization_2_layer_call_fn_350100rsvwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ä
6__inference_batch_normalization_2_layer_call_fn_350113rsvwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ì
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_350133tuxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ì
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_350151tuxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ä
6__inference_batch_normalization_3_layer_call_fn_350164tuxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ä
6__inference_batch_normalization_3_layer_call_fn_350177tuxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ð
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_350221~N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_350239~N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
6__inference_batch_normalization_4_layer_call_fn_350252~N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
6__inference_batch_normalization_4_layer_call_fn_350265~N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿò
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_350285N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ò
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_350303N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
6__inference_batch_normalization_5_layer_call_fn_350316N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÊ
6__inference_batch_normalization_5_layer_call_fn_350329N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_349917fgjkM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ê
O__inference_batch_normalization_layer_call_and_return_conditional_losses_349935fgjkM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Â
4__inference_batch_normalization_layer_call_fn_349948fgjkM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Â
4__inference_batch_normalization_layer_call_fn_349961fgjkM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¥
C__inference_dense_1_layer_call_and_return_conditional_losses_349680^JK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_1_layer_call_fn_349689QJK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_2_layer_call_and_return_conditional_losses_349727]XY0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_2_layer_call_fn_349736PXY0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_dense_layer_call_and_return_conditional_losses_349634^<=0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dense_layer_call_fn_349643Q<=0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÝ
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_346125R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ´
9__inference_global_average_pooling2d_layer_call_fn_346131wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_349752b¢

¢ 
ª " <
__inference_loss_fn_10_349862|¢

¢ 
ª " <
__inference_loss_fn_11_349873}¢

¢ 
ª " ;
__inference_loss_fn_1_349763c¢

¢ 
ª " ;
__inference_loss_fn_2_349774d¢

¢ 
ª " ;
__inference_loss_fn_3_349785e¢

¢ 
ª " ;
__inference_loss_fn_4_349796n¢

¢ 
ª " ;
__inference_loss_fn_5_349807o¢

¢ 
ª " ;
__inference_loss_fn_6_349818p¢

¢ 
ª " ;
__inference_loss_fn_7_349829q¢

¢ 
ª " ;
__inference_loss_fn_8_349840z¢

¢ 
ª " ;
__inference_loss_fn_9_349851{¢

¢ 
ª " î
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_345904R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_1_layer_call_fn_345910R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_345684R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_layer_call_fn_345690R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
Q__inference_monte_carlo_dropout_1_layer_call_and_return_conditional_losses_349701Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_monte_carlo_dropout_1_layer_call_fn_349706M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
O__inference_monte_carlo_dropout_layer_call_and_return_conditional_losses_349655Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_monte_carlo_dropout_layer_call_fn_349660M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÃ
K__inference_random_rotation_layer_call_and_return_conditional_losses_348930t=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¿
K__inference_random_rotation_layer_call_and_return_conditional_losses_348934p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_random_rotation_layer_call_fn_348941g=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ
0__inference_random_rotation_layer_call_fn_348946c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿÕ
$__inference_signature_wrapper_348044¬0bcfgjkdehilmnorsvwpqtuxyz{~|}<=JKXYE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ