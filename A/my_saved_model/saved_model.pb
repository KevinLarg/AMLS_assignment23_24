ИЄ
╣Ѕ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resourceѕ
ч
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48Њх
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Д
Adam/v/dense_69/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_69/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_69/bias
y
(Adam/v/dense_69/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_69/bias*
_output_shapes
:*
dtype0
Д
Adam/m/dense_69/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_69/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_69/bias
y
(Adam/m/dense_69/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_69/bias*
_output_shapes
:*
dtype0
▒
Adam/v/dense_69/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_69/kernel/*
dtype0*
shape
:@*'
shared_nameAdam/v/dense_69/kernel
Ђ
*Adam/v/dense_69/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_69/kernel*
_output_shapes

:@*
dtype0
▒
Adam/m/dense_69/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_69/kernel/*
dtype0*
shape
:@*'
shared_nameAdam/m/dense_69/kernel
Ђ
*Adam/m/dense_69/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_69/kernel*
_output_shapes

:@*
dtype0
Д
Adam/v/dense_68/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_68/bias/*
dtype0*
shape:@*%
shared_nameAdam/v/dense_68/bias
y
(Adam/v/dense_68/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_68/bias*
_output_shapes
:@*
dtype0
Д
Adam/m/dense_68/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_68/bias/*
dtype0*
shape:@*%
shared_nameAdam/m/dense_68/bias
y
(Adam/m/dense_68/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_68/bias*
_output_shapes
:@*
dtype0
▓
Adam/v/dense_68/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_68/kernel/*
dtype0*
shape:	└@*'
shared_nameAdam/v/dense_68/kernel
ѓ
*Adam/v/dense_68/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_68/kernel*
_output_shapes
:	└@*
dtype0
▓
Adam/m/dense_68/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_68/kernel/*
dtype0*
shape:	└@*'
shared_nameAdam/m/dense_68/kernel
ѓ
*Adam/m/dense_68/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_68/kernel*
_output_shapes
:	└@*
dtype0
Г
Adam/v/conv2d_111/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv2d_111/bias/*
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_111/bias
}
*Adam/v/conv2d_111/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_111/bias*
_output_shapes
:@*
dtype0
Г
Adam/m/conv2d_111/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv2d_111/bias/*
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_111/bias
}
*Adam/m/conv2d_111/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_111/bias*
_output_shapes
:@*
dtype0
┐
Adam/v/conv2d_111/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/v/conv2d_111/kernel/*
dtype0*
shape:@@*)
shared_nameAdam/v/conv2d_111/kernel
Ї
,Adam/v/conv2d_111/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_111/kernel*&
_output_shapes
:@@*
dtype0
┐
Adam/m/conv2d_111/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/m/conv2d_111/kernel/*
dtype0*
shape:@@*)
shared_nameAdam/m/conv2d_111/kernel
Ї
,Adam/m/conv2d_111/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_111/kernel*&
_output_shapes
:@@*
dtype0
н
#Adam/v/batch_normalization_112/betaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/v/batch_normalization_112/beta/*
dtype0*
shape:@*4
shared_name%#Adam/v/batch_normalization_112/beta
Ќ
7Adam/v/batch_normalization_112/beta/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_112/beta*
_output_shapes
:@*
dtype0
н
#Adam/m/batch_normalization_112/betaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/m/batch_normalization_112/beta/*
dtype0*
shape:@*4
shared_name%#Adam/m/batch_normalization_112/beta
Ќ
7Adam/m/batch_normalization_112/beta/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_112/beta*
_output_shapes
:@*
dtype0
О
$Adam/v/batch_normalization_112/gammaVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/v/batch_normalization_112/gamma/*
dtype0*
shape:@*5
shared_name&$Adam/v/batch_normalization_112/gamma
Ў
8Adam/v/batch_normalization_112/gamma/Read/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_112/gamma*
_output_shapes
:@*
dtype0
О
$Adam/m/batch_normalization_112/gammaVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/m/batch_normalization_112/gamma/*
dtype0*
shape:@*5
shared_name&$Adam/m/batch_normalization_112/gamma
Ў
8Adam/m/batch_normalization_112/gamma/Read/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_112/gamma*
_output_shapes
:@*
dtype0
Г
Adam/v/conv2d_110/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv2d_110/bias/*
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_110/bias
}
*Adam/v/conv2d_110/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_110/bias*
_output_shapes
:@*
dtype0
Г
Adam/m/conv2d_110/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv2d_110/bias/*
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_110/bias
}
*Adam/m/conv2d_110/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_110/bias*
_output_shapes
:@*
dtype0
┐
Adam/v/conv2d_110/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/v/conv2d_110/kernel/*
dtype0*
shape: @*)
shared_nameAdam/v/conv2d_110/kernel
Ї
,Adam/v/conv2d_110/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_110/kernel*&
_output_shapes
: @*
dtype0
┐
Adam/m/conv2d_110/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/m/conv2d_110/kernel/*
dtype0*
shape: @*)
shared_nameAdam/m/conv2d_110/kernel
Ї
,Adam/m/conv2d_110/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_110/kernel*&
_output_shapes
: @*
dtype0
н
#Adam/v/batch_normalization_111/betaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/v/batch_normalization_111/beta/*
dtype0*
shape: *4
shared_name%#Adam/v/batch_normalization_111/beta
Ќ
7Adam/v/batch_normalization_111/beta/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_111/beta*
_output_shapes
: *
dtype0
н
#Adam/m/batch_normalization_111/betaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/m/batch_normalization_111/beta/*
dtype0*
shape: *4
shared_name%#Adam/m/batch_normalization_111/beta
Ќ
7Adam/m/batch_normalization_111/beta/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_111/beta*
_output_shapes
: *
dtype0
О
$Adam/v/batch_normalization_111/gammaVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/v/batch_normalization_111/gamma/*
dtype0*
shape: *5
shared_name&$Adam/v/batch_normalization_111/gamma
Ў
8Adam/v/batch_normalization_111/gamma/Read/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_111/gamma*
_output_shapes
: *
dtype0
О
$Adam/m/batch_normalization_111/gammaVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/m/batch_normalization_111/gamma/*
dtype0*
shape: *5
shared_name&$Adam/m/batch_normalization_111/gamma
Ў
8Adam/m/batch_normalization_111/gamma/Read/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_111/gamma*
_output_shapes
: *
dtype0
Г
Adam/v/conv2d_109/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv2d_109/bias/*
dtype0*
shape: *'
shared_nameAdam/v/conv2d_109/bias
}
*Adam/v/conv2d_109/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_109/bias*
_output_shapes
: *
dtype0
Г
Adam/m/conv2d_109/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv2d_109/bias/*
dtype0*
shape: *'
shared_nameAdam/m/conv2d_109/bias
}
*Adam/m/conv2d_109/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_109/bias*
_output_shapes
: *
dtype0
┐
Adam/v/conv2d_109/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/v/conv2d_109/kernel/*
dtype0*
shape: *)
shared_nameAdam/v/conv2d_109/kernel
Ї
,Adam/v/conv2d_109/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_109/kernel*&
_output_shapes
: *
dtype0
┐
Adam/m/conv2d_109/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/m/conv2d_109/kernel/*
dtype0*
shape: *)
shared_nameAdam/m/conv2d_109/kernel
Ї
,Adam/m/conv2d_109/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_109/kernel*&
_output_shapes
: *
dtype0
н
#Adam/v/batch_normalization_110/betaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/v/batch_normalization_110/beta/*
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_110/beta
Ќ
7Adam/v/batch_normalization_110/beta/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_110/beta*
_output_shapes
:*
dtype0
н
#Adam/m/batch_normalization_110/betaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/m/batch_normalization_110/beta/*
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_110/beta
Ќ
7Adam/m/batch_normalization_110/beta/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_110/beta*
_output_shapes
:*
dtype0
О
$Adam/v/batch_normalization_110/gammaVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/v/batch_normalization_110/gamma/*
dtype0*
shape:*5
shared_name&$Adam/v/batch_normalization_110/gamma
Ў
8Adam/v/batch_normalization_110/gamma/Read/ReadVariableOpReadVariableOp$Adam/v/batch_normalization_110/gamma*
_output_shapes
:*
dtype0
О
$Adam/m/batch_normalization_110/gammaVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/m/batch_normalization_110/gamma/*
dtype0*
shape:*5
shared_name&$Adam/m/batch_normalization_110/gamma
Ў
8Adam/m/batch_normalization_110/gamma/Read/ReadVariableOpReadVariableOp$Adam/m/batch_normalization_110/gamma*
_output_shapes
:*
dtype0
ј
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
ѓ
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
њ
dense_69/biasVarHandleOp*
_output_shapes
: *

debug_namedense_69/bias/*
dtype0*
shape:*
shared_namedense_69/bias
k
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
_output_shapes
:*
dtype0
ю
dense_69/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_69/kernel/*
dtype0*
shape
:@* 
shared_namedense_69/kernel
s
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel*
_output_shapes

:@*
dtype0
њ
dense_68/biasVarHandleOp*
_output_shapes
: *

debug_namedense_68/bias/*
dtype0*
shape:@*
shared_namedense_68/bias
k
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes
:@*
dtype0
Ю
dense_68/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_68/kernel/*
dtype0*
shape:	└@* 
shared_namedense_68/kernel
t
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel*
_output_shapes
:	└@*
dtype0
ў
conv2d_111/biasVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_111/bias/*
dtype0*
shape:@* 
shared_nameconv2d_111/bias
o
#conv2d_111/bias/Read/ReadVariableOpReadVariableOpconv2d_111/bias*
_output_shapes
:@*
dtype0
ф
conv2d_111/kernelVarHandleOp*
_output_shapes
: *"

debug_nameconv2d_111/kernel/*
dtype0*
shape:@@*"
shared_nameconv2d_111/kernel

%conv2d_111/kernel/Read/ReadVariableOpReadVariableOpconv2d_111/kernel*&
_output_shapes
:@@*
dtype0
Я
'batch_normalization_112/moving_varianceVarHandleOp*
_output_shapes
: *8

debug_name*(batch_normalization_112/moving_variance/*
dtype0*
shape:@*8
shared_name)'batch_normalization_112/moving_variance
Ъ
;batch_normalization_112/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_112/moving_variance*
_output_shapes
:@*
dtype0
н
#batch_normalization_112/moving_meanVarHandleOp*
_output_shapes
: *4

debug_name&$batch_normalization_112/moving_mean/*
dtype0*
shape:@*4
shared_name%#batch_normalization_112/moving_mean
Ќ
7batch_normalization_112/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_112/moving_mean*
_output_shapes
:@*
dtype0
┐
batch_normalization_112/betaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_112/beta/*
dtype0*
shape:@*-
shared_namebatch_normalization_112/beta
Ѕ
0batch_normalization_112/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_112/beta*
_output_shapes
:@*
dtype0
┬
batch_normalization_112/gammaVarHandleOp*
_output_shapes
: *.

debug_name batch_normalization_112/gamma/*
dtype0*
shape:@*.
shared_namebatch_normalization_112/gamma
І
1batch_normalization_112/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_112/gamma*
_output_shapes
:@*
dtype0
ў
conv2d_110/biasVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_110/bias/*
dtype0*
shape:@* 
shared_nameconv2d_110/bias
o
#conv2d_110/bias/Read/ReadVariableOpReadVariableOpconv2d_110/bias*
_output_shapes
:@*
dtype0
ф
conv2d_110/kernelVarHandleOp*
_output_shapes
: *"

debug_nameconv2d_110/kernel/*
dtype0*
shape: @*"
shared_nameconv2d_110/kernel

%conv2d_110/kernel/Read/ReadVariableOpReadVariableOpconv2d_110/kernel*&
_output_shapes
: @*
dtype0
Я
'batch_normalization_111/moving_varianceVarHandleOp*
_output_shapes
: *8

debug_name*(batch_normalization_111/moving_variance/*
dtype0*
shape: *8
shared_name)'batch_normalization_111/moving_variance
Ъ
;batch_normalization_111/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_111/moving_variance*
_output_shapes
: *
dtype0
н
#batch_normalization_111/moving_meanVarHandleOp*
_output_shapes
: *4

debug_name&$batch_normalization_111/moving_mean/*
dtype0*
shape: *4
shared_name%#batch_normalization_111/moving_mean
Ќ
7batch_normalization_111/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_111/moving_mean*
_output_shapes
: *
dtype0
┐
batch_normalization_111/betaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_111/beta/*
dtype0*
shape: *-
shared_namebatch_normalization_111/beta
Ѕ
0batch_normalization_111/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_111/beta*
_output_shapes
: *
dtype0
┬
batch_normalization_111/gammaVarHandleOp*
_output_shapes
: *.

debug_name batch_normalization_111/gamma/*
dtype0*
shape: *.
shared_namebatch_normalization_111/gamma
І
1batch_normalization_111/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_111/gamma*
_output_shapes
: *
dtype0
ў
conv2d_109/biasVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_109/bias/*
dtype0*
shape: * 
shared_nameconv2d_109/bias
o
#conv2d_109/bias/Read/ReadVariableOpReadVariableOpconv2d_109/bias*
_output_shapes
: *
dtype0
ф
conv2d_109/kernelVarHandleOp*
_output_shapes
: *"

debug_nameconv2d_109/kernel/*
dtype0*
shape: *"
shared_nameconv2d_109/kernel

%conv2d_109/kernel/Read/ReadVariableOpReadVariableOpconv2d_109/kernel*&
_output_shapes
: *
dtype0
Я
'batch_normalization_110/moving_varianceVarHandleOp*
_output_shapes
: *8

debug_name*(batch_normalization_110/moving_variance/*
dtype0*
shape:*8
shared_name)'batch_normalization_110/moving_variance
Ъ
;batch_normalization_110/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_110/moving_variance*
_output_shapes
:*
dtype0
н
#batch_normalization_110/moving_meanVarHandleOp*
_output_shapes
: *4

debug_name&$batch_normalization_110/moving_mean/*
dtype0*
shape:*4
shared_name%#batch_normalization_110/moving_mean
Ќ
7batch_normalization_110/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_110/moving_mean*
_output_shapes
:*
dtype0
┐
batch_normalization_110/betaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_110/beta/*
dtype0*
shape:*-
shared_namebatch_normalization_110/beta
Ѕ
0batch_normalization_110/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_110/beta*
_output_shapes
:*
dtype0
┬
batch_normalization_110/gammaVarHandleOp*
_output_shapes
: *.

debug_name batch_normalization_110/gamma/*
dtype0*
shape:*.
shared_namebatch_normalization_110/gamma
І
1batch_normalization_110/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_110/gamma*
_output_shapes
:*
dtype0
а
-serving_default_batch_normalization_110_inputPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
┴
StatefulPartitionedCallStatefulPartitionedCall-serving_default_batch_normalization_110_inputbatch_normalization_110/gammabatch_normalization_110/beta#batch_normalization_110/moving_mean'batch_normalization_110/moving_varianceconv2d_109/kernelconv2d_109/biasbatch_normalization_111/gammabatch_normalization_111/beta#batch_normalization_111/moving_mean'batch_normalization_111/moving_varianceconv2d_110/kernelconv2d_110/biasbatch_normalization_112/gammabatch_normalization_112/beta#batch_normalization_112/moving_mean'batch_normalization_112/moving_varianceconv2d_111/kernelconv2d_111/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_583534

NoOpNoOp
Ќw
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*мv
value╚vB┼v BЙv
г
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
beta
moving_mean
moving_variance*
╚
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*
ј
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
Н
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5axis
	6gamma
7beta
8moving_mean
9moving_variance*
╚
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op*
ј
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
Н
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance*
╚
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias
 \_jit_compiled_convolution_op*
ј
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
д
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias*
д
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias*
ф
0
1
2
3
&4
'5
66
77
88
99
@10
A11
P12
Q13
R14
S15
Z16
[17
i18
j19
q20
r21*
z
0
1
&2
'3
64
75
@6
A7
P8
Q9
Z10
[11
i12
j13
q14
r15*
* 
░
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

xtrace_0
ytrace_1* 

ztrace_0
{trace_1* 
* 
ё
|
_variables
}_iterations
~_learning_rate
_index_dict
ђ
_momentums
Ђ_velocities
ѓ_update_step_xla*

Ѓserving_default* 
 
0
1
2
3*

0
1*
* 
ў
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Ѕtrace_0
іtrace_1
Іtrace_2
їtrace_3* 
:
Їtrace_0
јtrace_1
Јtrace_2
љtrace_3* 
* 
lf
VARIABLE_VALUEbatch_normalization_110/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_110/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_110/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE'batch_normalization_110/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 
ў
Љnon_trainable_variables
њlayers
Њmetrics
 ћlayer_regularization_losses
Ћlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

ќtrace_0* 

Ќtrace_0* 
a[
VARIABLE_VALUEconv2d_109/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_109/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

Юtrace_0* 

ъtrace_0* 
 
60
71
82
93*

60
71*
* 
ў
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

цtrace_0
Цtrace_1* 

дtrace_0
Дtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_111/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_111/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_111/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE'batch_normalization_111/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

@0
A1*
* 
ў
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

Гtrace_0* 

«trace_0* 
a[
VARIABLE_VALUEconv2d_110/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_110/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
»non_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

┤trace_0* 

хtrace_0* 
 
P0
Q1
R2
S3*

P0
Q1*
* 
ў
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

╗trace_0
╝trace_1* 

йtrace_0
Йtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_112/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_112/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#batch_normalization_112/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE'batch_normalization_112/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

Z0
[1*
* 
ў
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

─trace_0* 

┼trace_0* 
a[
VARIABLE_VALUEconv2d_111/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_111/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
кnon_trainable_variables
Кlayers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

╦trace_0* 

╠trace_0* 

i0
j1*

i0
j1*
* 
ў
═non_trainable_variables
╬layers
¤metrics
 лlayer_regularization_losses
Лlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

мtrace_0* 

Мtrace_0* 
_Y
VARIABLE_VALUEdense_68/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_68/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

q0
r1*

q0
r1*
* 
ў
нnon_trainable_variables
Нlayers
оmetrics
 Оlayer_regularization_losses
пlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

┘trace_0* 

┌trace_0* 
_Y
VARIABLE_VALUEdense_69/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_69/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
.
0
1
82
93
R4
S5*
R
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
10*

█0
▄1*
* 
* 
* 
* 
* 
* 
б
}0
П1
я2
▀3
Я4
р5
Р6
с7
С8
т9
Т10
у11
У12
ж13
Ж14
в15
В16
ь17
Ь18
№19
­20
ы21
Ы22
з23
З24
ш25
Ш26
э27
Э28
щ29
Щ30
ч31
Ч32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
і
П0
▀1
р2
с3
т4
у5
ж6
в7
ь8
№9
ы10
з11
ш12
э13
щ14
ч15*
і
я0
Я1
Р2
С3
Т4
У5
Ж6
В7
Ь8
­9
Ы10
З11
Ш12
Э13
Щ14
Ч15*
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

80
91*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

R0
S1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
§	variables
■	keras_api

 total

ђcount*
M
Ђ	variables
ѓ	keras_api

Ѓtotal

ёcount
Ё
_fn_kwargs*
oi
VARIABLE_VALUE$Adam/m/batch_normalization_110/gamma1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/batch_normalization_110/gamma1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/batch_normalization_110/beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/batch_normalization_110/beta1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_109/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_109/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_109/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_109/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/batch_normalization_111/gamma1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/batch_normalization_111/gamma2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_111/beta2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_111/beta2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/conv2d_110/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/conv2d_110/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_110/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_110/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/batch_normalization_112/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/batch_normalization_112/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_112/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_112/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/conv2d_111/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/conv2d_111/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_111/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_111/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_68/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_68/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_68/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_68/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_69/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_69/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_69/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_69/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*

 0
ђ1*

§	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ѓ0
ё1*

Ђ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ё
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebatch_normalization_110/gammabatch_normalization_110/beta#batch_normalization_110/moving_mean'batch_normalization_110/moving_varianceconv2d_109/kernelconv2d_109/biasbatch_normalization_111/gammabatch_normalization_111/beta#batch_normalization_111/moving_mean'batch_normalization_111/moving_varianceconv2d_110/kernelconv2d_110/biasbatch_normalization_112/gammabatch_normalization_112/beta#batch_normalization_112/moving_mean'batch_normalization_112/moving_varianceconv2d_111/kernelconv2d_111/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/bias	iterationlearning_rate$Adam/m/batch_normalization_110/gamma$Adam/v/batch_normalization_110/gamma#Adam/m/batch_normalization_110/beta#Adam/v/batch_normalization_110/betaAdam/m/conv2d_109/kernelAdam/v/conv2d_109/kernelAdam/m/conv2d_109/biasAdam/v/conv2d_109/bias$Adam/m/batch_normalization_111/gamma$Adam/v/batch_normalization_111/gamma#Adam/m/batch_normalization_111/beta#Adam/v/batch_normalization_111/betaAdam/m/conv2d_110/kernelAdam/v/conv2d_110/kernelAdam/m/conv2d_110/biasAdam/v/conv2d_110/bias$Adam/m/batch_normalization_112/gamma$Adam/v/batch_normalization_112/gamma#Adam/m/batch_normalization_112/beta#Adam/v/batch_normalization_112/betaAdam/m/conv2d_111/kernelAdam/v/conv2d_111/kernelAdam/m/conv2d_111/biasAdam/v/conv2d_111/biasAdam/m/dense_68/kernelAdam/v/dense_68/kernelAdam/m/dense_68/biasAdam/v/dense_68/biasAdam/m/dense_69/kernelAdam/v/dense_69/kernelAdam/m/dense_69/biasAdam/v/dense_69/biastotal_1count_1totalcountConst*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__traced_save_584297
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_110/gammabatch_normalization_110/beta#batch_normalization_110/moving_mean'batch_normalization_110/moving_varianceconv2d_109/kernelconv2d_109/biasbatch_normalization_111/gammabatch_normalization_111/beta#batch_normalization_111/moving_mean'batch_normalization_111/moving_varianceconv2d_110/kernelconv2d_110/biasbatch_normalization_112/gammabatch_normalization_112/beta#batch_normalization_112/moving_mean'batch_normalization_112/moving_varianceconv2d_111/kernelconv2d_111/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/bias	iterationlearning_rate$Adam/m/batch_normalization_110/gamma$Adam/v/batch_normalization_110/gamma#Adam/m/batch_normalization_110/beta#Adam/v/batch_normalization_110/betaAdam/m/conv2d_109/kernelAdam/v/conv2d_109/kernelAdam/m/conv2d_109/biasAdam/v/conv2d_109/bias$Adam/m/batch_normalization_111/gamma$Adam/v/batch_normalization_111/gamma#Adam/m/batch_normalization_111/beta#Adam/v/batch_normalization_111/betaAdam/m/conv2d_110/kernelAdam/v/conv2d_110/kernelAdam/m/conv2d_110/biasAdam/v/conv2d_110/bias$Adam/m/batch_normalization_112/gamma$Adam/v/batch_normalization_112/gamma#Adam/m/batch_normalization_112/beta#Adam/v/batch_normalization_112/betaAdam/m/conv2d_111/kernelAdam/v/conv2d_111/kernelAdam/m/conv2d_111/biasAdam/v/conv2d_111/biasAdam/m/dense_68/kernelAdam/v/dense_68/kernelAdam/m/dense_68/biasAdam/v/dense_68/biasAdam/m/dense_69/kernelAdam/v/dense_69/kernelAdam/m/dense_69/biasAdam/v/dense_69/biastotal_1count_1totalcount*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_584486яђ
║
M
1__inference_max_pooling2d_77_layer_call_fn_583685

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_582952Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Х
 
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583132

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ЛB
р

I__inference_sequential_34_layer_call_and_return_conditional_losses_583308!
batch_normalization_110_input,
batch_normalization_110_583252:,
batch_normalization_110_583254:,
batch_normalization_110_583256:,
batch_normalization_110_583258:+
conv2d_109_583261: 
conv2d_109_583263: ,
batch_normalization_111_583267: ,
batch_normalization_111_583269: ,
batch_normalization_111_583271: ,
batch_normalization_111_583273: +
conv2d_110_583276: @
conv2d_110_583278:@,
batch_normalization_112_583282:@,
batch_normalization_112_583284:@,
batch_normalization_112_583286:@,
batch_normalization_112_583288:@+
conv2d_111_583291:@@
conv2d_111_583293:@"
dense_68_583297:	└@
dense_68_583299:@!
dense_69_583302:@
dense_69_583304:
identityѕб/batch_normalization_110/StatefulPartitionedCallб/batch_normalization_111/StatefulPartitionedCallб/batch_normalization_112/StatefulPartitionedCallб"conv2d_109/StatefulPartitionedCallб"conv2d_110/StatefulPartitionedCallб"conv2d_111/StatefulPartitionedCallб dense_68/StatefulPartitionedCallб dense_69/StatefulPartitionedCallЈ
/batch_normalization_110/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_110_inputbatch_normalization_110_583252batch_normalization_110_583254batch_normalization_110_583256batch_normalization_110_583258*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583251▓
"conv2d_109/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_110/StatefulPartitionedCall:output:0conv2d_109_583261conv2d_109_583263*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583132ш
 max_pooling2d_77/PartitionedCallPartitionedCall+conv2d_109/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_582952Џ
/batch_normalization_111/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_77/PartitionedCall:output:0batch_normalization_111_583267batch_normalization_111_583269batch_normalization_111_583271batch_normalization_111_583273*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_582993▓
"conv2d_110/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_111/StatefulPartitionedCall:output:0conv2d_110_583276conv2d_110_583278*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583158ш
 max_pooling2d_78/PartitionedCallPartitionedCall+conv2d_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583024Џ
/batch_normalization_112/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_78/PartitionedCall:output:0batch_normalization_112_583282batch_normalization_112_583284batch_normalization_112_583286batch_normalization_112_583288*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583065▓
"conv2d_111/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_112/StatefulPartitionedCall:output:0conv2d_111_583291conv2d_111_583293*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583184Р
flatten_34/PartitionedCallPartitionedCall+conv2d_111/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_34_layer_call_and_return_conditional_losses_583195Ї
 dense_68/StatefulPartitionedCallStatefulPartitionedCall#flatten_34/PartitionedCall:output:0dense_68_583297dense_68_583299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_583207Њ
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_583302dense_69_583304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_583223x
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ь
NoOpNoOp0^batch_normalization_110/StatefulPartitionedCall0^batch_normalization_111/StatefulPartitionedCall0^batch_normalization_112/StatefulPartitionedCall#^conv2d_109/StatefulPartitionedCall#^conv2d_110/StatefulPartitionedCall#^conv2d_111/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_110/StatefulPartitionedCall/batch_normalization_110/StatefulPartitionedCall2b
/batch_normalization_111/StatefulPartitionedCall/batch_normalization_111/StatefulPartitionedCall2b
/batch_normalization_112/StatefulPartitionedCall/batch_normalization_112/StatefulPartitionedCall2H
"conv2d_109/StatefulPartitionedCall"conv2d_109/StatefulPartitionedCall2H
"conv2d_110/StatefulPartitionedCall"conv2d_110/StatefulPartitionedCall2H
"conv2d_111/StatefulPartitionedCall"conv2d_111/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:&"
 
_user_specified_name583304:&"
 
_user_specified_name583302:&"
 
_user_specified_name583299:&"
 
_user_specified_name583297:&"
 
_user_specified_name583293:&"
 
_user_specified_name583291:&"
 
_user_specified_name583288:&"
 
_user_specified_name583286:&"
 
_user_specified_name583284:&"
 
_user_specified_name583282:&"
 
_user_specified_name583278:&"
 
_user_specified_name583276:&
"
 
_user_specified_name583273:&	"
 
_user_specified_name583271:&"
 
_user_specified_name583269:&"
 
_user_specified_name583267:&"
 
_user_specified_name583263:&"
 
_user_specified_name583261:&"
 
_user_specified_name583258:&"
 
_user_specified_name583256:&"
 
_user_specified_name583254:&"
 
_user_specified_name583252:n j
/
_output_shapes
:         
7
_user_specified_namebatch_normalization_110_input
к
┌
.__inference_sequential_34_layer_call_fn_583357!
batch_normalization_110_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_110_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_583230o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583353:&"
 
_user_specified_name583351:&"
 
_user_specified_name583349:&"
 
_user_specified_name583347:&"
 
_user_specified_name583345:&"
 
_user_specified_name583343:&"
 
_user_specified_name583341:&"
 
_user_specified_name583339:&"
 
_user_specified_name583337:&"
 
_user_specified_name583335:&"
 
_user_specified_name583333:&"
 
_user_specified_name583331:&
"
 
_user_specified_name583329:&	"
 
_user_specified_name583327:&"
 
_user_specified_name583325:&"
 
_user_specified_name583323:&"
 
_user_specified_name583321:&"
 
_user_specified_name583319:&"
 
_user_specified_name583317:&"
 
_user_specified_name583315:&"
 
_user_specified_name583313:&"
 
_user_specified_name583311:n j
/
_output_shapes
:         
7
_user_specified_namebatch_normalization_110_input
ї
┬
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_582903

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Ь
ќ
)__inference_dense_69_layer_call_fn_583904

inputs
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_583223o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583900:&"
 
_user_specified_name583898:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ц
┬
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583641

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0к
FusedBatchNormV3FusedBatchNormV3Cast:y:0ReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         ░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Х
 
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583680

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
м
ъ
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583752

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
║
M
1__inference_max_pooling2d_78_layer_call_fn_583777

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583024Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
м
ъ
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_582993

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
МЏ
ь'
"__inference__traced_restore_584486
file_prefix<
.assignvariableop_batch_normalization_110_gamma:=
/assignvariableop_1_batch_normalization_110_beta:D
6assignvariableop_2_batch_normalization_110_moving_mean:H
:assignvariableop_3_batch_normalization_110_moving_variance:>
$assignvariableop_4_conv2d_109_kernel: 0
"assignvariableop_5_conv2d_109_bias: >
0assignvariableop_6_batch_normalization_111_gamma: =
/assignvariableop_7_batch_normalization_111_beta: D
6assignvariableop_8_batch_normalization_111_moving_mean: H
:assignvariableop_9_batch_normalization_111_moving_variance: ?
%assignvariableop_10_conv2d_110_kernel: @1
#assignvariableop_11_conv2d_110_bias:@?
1assignvariableop_12_batch_normalization_112_gamma:@>
0assignvariableop_13_batch_normalization_112_beta:@E
7assignvariableop_14_batch_normalization_112_moving_mean:@I
;assignvariableop_15_batch_normalization_112_moving_variance:@?
%assignvariableop_16_conv2d_111_kernel:@@1
#assignvariableop_17_conv2d_111_bias:@6
#assignvariableop_18_dense_68_kernel:	└@/
!assignvariableop_19_dense_68_bias:@5
#assignvariableop_20_dense_69_kernel:@/
!assignvariableop_21_dense_69_bias:'
assignvariableop_22_iteration:	 +
!assignvariableop_23_learning_rate: F
8assignvariableop_24_adam_m_batch_normalization_110_gamma:F
8assignvariableop_25_adam_v_batch_normalization_110_gamma:E
7assignvariableop_26_adam_m_batch_normalization_110_beta:E
7assignvariableop_27_adam_v_batch_normalization_110_beta:F
,assignvariableop_28_adam_m_conv2d_109_kernel: F
,assignvariableop_29_adam_v_conv2d_109_kernel: 8
*assignvariableop_30_adam_m_conv2d_109_bias: 8
*assignvariableop_31_adam_v_conv2d_109_bias: F
8assignvariableop_32_adam_m_batch_normalization_111_gamma: F
8assignvariableop_33_adam_v_batch_normalization_111_gamma: E
7assignvariableop_34_adam_m_batch_normalization_111_beta: E
7assignvariableop_35_adam_v_batch_normalization_111_beta: F
,assignvariableop_36_adam_m_conv2d_110_kernel: @F
,assignvariableop_37_adam_v_conv2d_110_kernel: @8
*assignvariableop_38_adam_m_conv2d_110_bias:@8
*assignvariableop_39_adam_v_conv2d_110_bias:@F
8assignvariableop_40_adam_m_batch_normalization_112_gamma:@F
8assignvariableop_41_adam_v_batch_normalization_112_gamma:@E
7assignvariableop_42_adam_m_batch_normalization_112_beta:@E
7assignvariableop_43_adam_v_batch_normalization_112_beta:@F
,assignvariableop_44_adam_m_conv2d_111_kernel:@@F
,assignvariableop_45_adam_v_conv2d_111_kernel:@@8
*assignvariableop_46_adam_m_conv2d_111_bias:@8
*assignvariableop_47_adam_v_conv2d_111_bias:@=
*assignvariableop_48_adam_m_dense_68_kernel:	└@=
*assignvariableop_49_adam_v_dense_68_kernel:	└@6
(assignvariableop_50_adam_m_dense_68_bias:@6
(assignvariableop_51_adam_v_dense_68_bias:@<
*assignvariableop_52_adam_m_dense_69_kernel:@<
*assignvariableop_53_adam_v_dense_69_kernel:@6
(assignvariableop_54_adam_m_dense_69_bias:6
(assignvariableop_55_adam_v_dense_69_bias:%
assignvariableop_56_total_1: %
assignvariableop_57_count_1: #
assignvariableop_58_total: #
assignvariableop_59_count: 
identity_61ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*л
valueкB├=B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHь
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*Ј
valueЁBѓ=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B м
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*і
_output_shapesэ
З:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOpAssignVariableOp.assignvariableop_batch_normalization_110_gammaIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_110_betaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_2AssignVariableOp6assignvariableop_2_batch_normalization_110_moving_meanIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_3AssignVariableOp:assignvariableop_3_batch_normalization_110_moving_varianceIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_109_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_109_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_6AssignVariableOp0assignvariableop_6_batch_normalization_111_gammaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:к
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_111_betaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_8AssignVariableOp6assignvariableop_8_batch_normalization_111_moving_meanIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_9AssignVariableOp:assignvariableop_9_batch_normalization_111_moving_varianceIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv2d_110_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_110_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_12AssignVariableOp1assignvariableop_12_batch_normalization_112_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╔
AssignVariableOp_13AssignVariableOp0assignvariableop_13_batch_normalization_112_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_14AssignVariableOp7assignvariableop_14_batch_normalization_112_moving_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_15AssignVariableOp;assignvariableop_15_batch_normalization_112_moving_varianceIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_16AssignVariableOp%assignvariableop_16_conv2d_111_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv2d_111_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_68_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_68_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_69_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_69_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:Х
AssignVariableOp_22AssignVariableOpassignvariableop_22_iterationIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_24AssignVariableOp8assignvariableop_24_adam_m_batch_normalization_110_gammaIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_v_batch_normalization_110_gammaIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_26AssignVariableOp7assignvariableop_26_adam_m_batch_normalization_110_betaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_v_batch_normalization_110_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_m_conv2d_109_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_v_conv2d_109_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_conv2d_109_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_conv2d_109_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_32AssignVariableOp8assignvariableop_32_adam_m_batch_normalization_111_gammaIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_33AssignVariableOp8assignvariableop_33_adam_v_batch_normalization_111_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_m_batch_normalization_111_betaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_v_batch_normalization_111_betaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_m_conv2d_110_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_v_conv2d_110_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_conv2d_110_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_conv2d_110_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_40AssignVariableOp8assignvariableop_40_adam_m_batch_normalization_112_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_41AssignVariableOp8assignvariableop_41_adam_v_batch_normalization_112_gammaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adam_m_batch_normalization_112_betaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:л
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_v_batch_normalization_112_betaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_m_conv2d_111_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_v_conv2d_111_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_m_conv2d_111_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_v_conv2d_111_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_dense_68_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_dense_68_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_dense_68_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_dense_68_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_m_dense_69_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_v_dense_69_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_m_dense_69_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_v_dense_69_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_1Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_1Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_58AssignVariableOpassignvariableop_58_totalIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_59AssignVariableOpassignvariableop_59_countIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 э

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: └

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*Ї
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
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
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%<!

_user_specified_namecount:%;!

_user_specified_nametotal:':#
!
_user_specified_name	count_1:'9#
!
_user_specified_name	total_1:480
.
_user_specified_nameAdam/v/dense_69/bias:470
.
_user_specified_nameAdam/m/dense_69/bias:662
0
_user_specified_nameAdam/v/dense_69/kernel:652
0
_user_specified_nameAdam/m/dense_69/kernel:440
.
_user_specified_nameAdam/v/dense_68/bias:430
.
_user_specified_nameAdam/m/dense_68/bias:622
0
_user_specified_nameAdam/v/dense_68/kernel:612
0
_user_specified_nameAdam/m/dense_68/kernel:602
0
_user_specified_nameAdam/v/conv2d_111/bias:6/2
0
_user_specified_nameAdam/m/conv2d_111/bias:8.4
2
_user_specified_nameAdam/v/conv2d_111/kernel:8-4
2
_user_specified_nameAdam/m/conv2d_111/kernel:C,?
=
_user_specified_name%#Adam/v/batch_normalization_112/beta:C+?
=
_user_specified_name%#Adam/m/batch_normalization_112/beta:D*@
>
_user_specified_name&$Adam/v/batch_normalization_112/gamma:D)@
>
_user_specified_name&$Adam/m/batch_normalization_112/gamma:6(2
0
_user_specified_nameAdam/v/conv2d_110/bias:6'2
0
_user_specified_nameAdam/m/conv2d_110/bias:8&4
2
_user_specified_nameAdam/v/conv2d_110/kernel:8%4
2
_user_specified_nameAdam/m/conv2d_110/kernel:C$?
=
_user_specified_name%#Adam/v/batch_normalization_111/beta:C#?
=
_user_specified_name%#Adam/m/batch_normalization_111/beta:D"@
>
_user_specified_name&$Adam/v/batch_normalization_111/gamma:D!@
>
_user_specified_name&$Adam/m/batch_normalization_111/gamma:6 2
0
_user_specified_nameAdam/v/conv2d_109/bias:62
0
_user_specified_nameAdam/m/conv2d_109/bias:84
2
_user_specified_nameAdam/v/conv2d_109/kernel:84
2
_user_specified_nameAdam/m/conv2d_109/kernel:C?
=
_user_specified_name%#Adam/v/batch_normalization_110/beta:C?
=
_user_specified_name%#Adam/m/batch_normalization_110/beta:D@
>
_user_specified_name&$Adam/v/batch_normalization_110/gamma:D@
>
_user_specified_name&$Adam/m/batch_normalization_110/gamma:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_69/bias:/+
)
_user_specified_namedense_69/kernel:-)
'
_user_specified_namedense_68/bias:/+
)
_user_specified_namedense_68/kernel:/+
)
_user_specified_nameconv2d_111/bias:1-
+
_user_specified_nameconv2d_111/kernel:GC
A
_user_specified_name)'batch_normalization_112/moving_variance:C?
=
_user_specified_name%#batch_normalization_112/moving_mean:<8
6
_user_specified_namebatch_normalization_112/beta:=9
7
_user_specified_namebatch_normalization_112/gamma:/+
)
_user_specified_nameconv2d_110/bias:1-
+
_user_specified_nameconv2d_110/kernel:G
C
A
_user_specified_name)'batch_normalization_111/moving_variance:C	?
=
_user_specified_name%#batch_normalization_111/moving_mean:<8
6
_user_specified_namebatch_normalization_111/beta:=9
7
_user_specified_namebatch_normalization_111/gamma:/+
)
_user_specified_nameconv2d_109/bias:1-
+
_user_specified_nameconv2d_109/kernel:GC
A
_user_specified_name)'batch_normalization_110/moving_variance:C?
=
_user_specified_name%#batch_normalization_110/moving_mean:<8
6
_user_specified_namebatch_normalization_110/beta:=9
7
_user_specified_namebatch_normalization_110/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ј

М
8__inference_batch_normalization_112_layer_call_fn_583795

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583047Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583791:&"
 
_user_specified_name583789:&"
 
_user_specified_name583787:&"
 
_user_specified_name583785:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ї
┬
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583734

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ї
┬
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583604

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╠
┌
.__inference_sequential_34_layer_call_fn_583406!
batch_normalization_110_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:
identityѕбStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_110_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_34_layer_call_and_return_conditional_losses_583308o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583402:&"
 
_user_specified_name583400:&"
 
_user_specified_name583398:&"
 
_user_specified_name583396:&"
 
_user_specified_name583394:&"
 
_user_specified_name583392:&"
 
_user_specified_name583390:&"
 
_user_specified_name583388:&"
 
_user_specified_name583386:&"
 
_user_specified_name583384:&"
 
_user_specified_name583382:&"
 
_user_specified_name583380:&
"
 
_user_specified_name583378:&	"
 
_user_specified_name583376:&"
 
_user_specified_name583374:&"
 
_user_specified_name583372:&"
 
_user_specified_name583370:&"
 
_user_specified_name583368:&"
 
_user_specified_name583366:&"
 
_user_specified_name583364:&"
 
_user_specified_name583362:&"
 
_user_specified_name583360:n j
/
_output_shapes
:         
7
_user_specified_namebatch_normalization_110_input
џ
л
$__inference_signature_wrapper_583534!
batch_normalization_110_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: @

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:	└@

unknown_18:@

unknown_19:@

unknown_20:
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_110_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_582885o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583530:&"
 
_user_specified_name583528:&"
 
_user_specified_name583526:&"
 
_user_specified_name583524:&"
 
_user_specified_name583522:&"
 
_user_specified_name583520:&"
 
_user_specified_name583518:&"
 
_user_specified_name583516:&"
 
_user_specified_name583514:&"
 
_user_specified_name583512:&"
 
_user_specified_name583510:&"
 
_user_specified_name583508:&
"
 
_user_specified_name583506:&	"
 
_user_specified_name583504:&"
 
_user_specified_name583502:&"
 
_user_specified_name583500:&"
 
_user_specified_name583498:&"
 
_user_specified_name583496:&"
 
_user_specified_name583494:&"
 
_user_specified_name583492:&"
 
_user_specified_name583490:&"
 
_user_specified_name583488:n j
/
_output_shapes
:         
7
_user_specified_namebatch_normalization_110_input
Љ

М
8__inference_batch_normalization_111_layer_call_fn_583716

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_582993Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583712:&"
 
_user_specified_name583710:&"
 
_user_specified_name583708:&"
 
_user_specified_name583706:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
║╬
▓9
__inference__traced_save_584297
file_prefixB
4read_disablecopyonread_batch_normalization_110_gamma:C
5read_1_disablecopyonread_batch_normalization_110_beta:J
<read_2_disablecopyonread_batch_normalization_110_moving_mean:N
@read_3_disablecopyonread_batch_normalization_110_moving_variance:D
*read_4_disablecopyonread_conv2d_109_kernel: 6
(read_5_disablecopyonread_conv2d_109_bias: D
6read_6_disablecopyonread_batch_normalization_111_gamma: C
5read_7_disablecopyonread_batch_normalization_111_beta: J
<read_8_disablecopyonread_batch_normalization_111_moving_mean: N
@read_9_disablecopyonread_batch_normalization_111_moving_variance: E
+read_10_disablecopyonread_conv2d_110_kernel: @7
)read_11_disablecopyonread_conv2d_110_bias:@E
7read_12_disablecopyonread_batch_normalization_112_gamma:@D
6read_13_disablecopyonread_batch_normalization_112_beta:@K
=read_14_disablecopyonread_batch_normalization_112_moving_mean:@O
Aread_15_disablecopyonread_batch_normalization_112_moving_variance:@E
+read_16_disablecopyonread_conv2d_111_kernel:@@7
)read_17_disablecopyonread_conv2d_111_bias:@<
)read_18_disablecopyonread_dense_68_kernel:	└@5
'read_19_disablecopyonread_dense_68_bias:@;
)read_20_disablecopyonread_dense_69_kernel:@5
'read_21_disablecopyonread_dense_69_bias:-
#read_22_disablecopyonread_iteration:	 1
'read_23_disablecopyonread_learning_rate: L
>read_24_disablecopyonread_adam_m_batch_normalization_110_gamma:L
>read_25_disablecopyonread_adam_v_batch_normalization_110_gamma:K
=read_26_disablecopyonread_adam_m_batch_normalization_110_beta:K
=read_27_disablecopyonread_adam_v_batch_normalization_110_beta:L
2read_28_disablecopyonread_adam_m_conv2d_109_kernel: L
2read_29_disablecopyonread_adam_v_conv2d_109_kernel: >
0read_30_disablecopyonread_adam_m_conv2d_109_bias: >
0read_31_disablecopyonread_adam_v_conv2d_109_bias: L
>read_32_disablecopyonread_adam_m_batch_normalization_111_gamma: L
>read_33_disablecopyonread_adam_v_batch_normalization_111_gamma: K
=read_34_disablecopyonread_adam_m_batch_normalization_111_beta: K
=read_35_disablecopyonread_adam_v_batch_normalization_111_beta: L
2read_36_disablecopyonread_adam_m_conv2d_110_kernel: @L
2read_37_disablecopyonread_adam_v_conv2d_110_kernel: @>
0read_38_disablecopyonread_adam_m_conv2d_110_bias:@>
0read_39_disablecopyonread_adam_v_conv2d_110_bias:@L
>read_40_disablecopyonread_adam_m_batch_normalization_112_gamma:@L
>read_41_disablecopyonread_adam_v_batch_normalization_112_gamma:@K
=read_42_disablecopyonread_adam_m_batch_normalization_112_beta:@K
=read_43_disablecopyonread_adam_v_batch_normalization_112_beta:@L
2read_44_disablecopyonread_adam_m_conv2d_111_kernel:@@L
2read_45_disablecopyonread_adam_v_conv2d_111_kernel:@@>
0read_46_disablecopyonread_adam_m_conv2d_111_bias:@>
0read_47_disablecopyonread_adam_v_conv2d_111_bias:@C
0read_48_disablecopyonread_adam_m_dense_68_kernel:	└@C
0read_49_disablecopyonread_adam_v_dense_68_kernel:	└@<
.read_50_disablecopyonread_adam_m_dense_68_bias:@<
.read_51_disablecopyonread_adam_v_dense_68_bias:@B
0read_52_disablecopyonread_adam_m_dense_69_kernel:@B
0read_53_disablecopyonread_adam_v_dense_69_kernel:@<
.read_54_disablecopyonread_adam_m_dense_69_bias:<
.read_55_disablecopyonread_adam_v_dense_69_bias:+
!read_56_disablecopyonread_total_1: +
!read_57_disablecopyonread_count_1: )
read_58_disablecopyonread_total: )
read_59_disablecopyonread_count: 
savev2_const
identity_121ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_18/DisableCopyOnReadбRead_18/ReadVariableOpбRead_19/DisableCopyOnReadбRead_19/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_20/DisableCopyOnReadбRead_20/ReadVariableOpбRead_21/DisableCopyOnReadбRead_21/ReadVariableOpбRead_22/DisableCopyOnReadбRead_22/ReadVariableOpбRead_23/DisableCopyOnReadбRead_23/ReadVariableOpбRead_24/DisableCopyOnReadбRead_24/ReadVariableOpбRead_25/DisableCopyOnReadбRead_25/ReadVariableOpбRead_26/DisableCopyOnReadбRead_26/ReadVariableOpбRead_27/DisableCopyOnReadбRead_27/ReadVariableOpбRead_28/DisableCopyOnReadбRead_28/ReadVariableOpбRead_29/DisableCopyOnReadбRead_29/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_30/DisableCopyOnReadбRead_30/ReadVariableOpбRead_31/DisableCopyOnReadбRead_31/ReadVariableOpбRead_32/DisableCopyOnReadбRead_32/ReadVariableOpбRead_33/DisableCopyOnReadбRead_33/ReadVariableOpбRead_34/DisableCopyOnReadбRead_34/ReadVariableOpбRead_35/DisableCopyOnReadбRead_35/ReadVariableOpбRead_36/DisableCopyOnReadбRead_36/ReadVariableOpбRead_37/DisableCopyOnReadбRead_37/ReadVariableOpбRead_38/DisableCopyOnReadбRead_38/ReadVariableOpбRead_39/DisableCopyOnReadбRead_39/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_40/DisableCopyOnReadбRead_40/ReadVariableOpбRead_41/DisableCopyOnReadбRead_41/ReadVariableOpбRead_42/DisableCopyOnReadбRead_42/ReadVariableOpбRead_43/DisableCopyOnReadбRead_43/ReadVariableOpбRead_44/DisableCopyOnReadбRead_44/ReadVariableOpбRead_45/DisableCopyOnReadбRead_45/ReadVariableOpбRead_46/DisableCopyOnReadбRead_46/ReadVariableOpбRead_47/DisableCopyOnReadбRead_47/ReadVariableOpбRead_48/DisableCopyOnReadбRead_48/ReadVariableOpбRead_49/DisableCopyOnReadбRead_49/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_50/DisableCopyOnReadбRead_50/ReadVariableOpбRead_51/DisableCopyOnReadбRead_51/ReadVariableOpбRead_52/DisableCopyOnReadбRead_52/ReadVariableOpбRead_53/DisableCopyOnReadбRead_53/ReadVariableOpбRead_54/DisableCopyOnReadбRead_54/ReadVariableOpбRead_55/DisableCopyOnReadбRead_55/ReadVariableOpбRead_56/DisableCopyOnReadбRead_56/ReadVariableOpбRead_57/DisableCopyOnReadбRead_57/ReadVariableOpбRead_58/DisableCopyOnReadбRead_58/ReadVariableOpбRead_59/DisableCopyOnReadбRead_59/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: є
Read/DisableCopyOnReadDisableCopyOnRead4read_disablecopyonread_batch_normalization_110_gamma"/device:CPU:0*
_output_shapes
 г
Read/ReadVariableOpReadVariableOp4read_disablecopyonread_batch_normalization_110_gamma^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:Ѕ
Read_1/DisableCopyOnReadDisableCopyOnRead5read_1_disablecopyonread_batch_normalization_110_beta"/device:CPU:0*
_output_shapes
 ▒
Read_1/ReadVariableOpReadVariableOp5read_1_disablecopyonread_batch_normalization_110_beta^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:љ
Read_2/DisableCopyOnReadDisableCopyOnRead<read_2_disablecopyonread_batch_normalization_110_moving_mean"/device:CPU:0*
_output_shapes
 И
Read_2/ReadVariableOpReadVariableOp<read_2_disablecopyonread_batch_normalization_110_moving_mean^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:ћ
Read_3/DisableCopyOnReadDisableCopyOnRead@read_3_disablecopyonread_batch_normalization_110_moving_variance"/device:CPU:0*
_output_shapes
 ╝
Read_3/ReadVariableOpReadVariableOp@read_3_disablecopyonread_batch_normalization_110_moving_variance^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_4/DisableCopyOnReadDisableCopyOnRead*read_4_disablecopyonread_conv2d_109_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_4/ReadVariableOpReadVariableOp*read_4_disablecopyonread_conv2d_109_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: |
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_conv2d_109_bias"/device:CPU:0*
_output_shapes
 ц
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_conv2d_109_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: і
Read_6/DisableCopyOnReadDisableCopyOnRead6read_6_disablecopyonread_batch_normalization_111_gamma"/device:CPU:0*
_output_shapes
 ▓
Read_6/ReadVariableOpReadVariableOp6read_6_disablecopyonread_batch_normalization_111_gamma^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: Ѕ
Read_7/DisableCopyOnReadDisableCopyOnRead5read_7_disablecopyonread_batch_normalization_111_beta"/device:CPU:0*
_output_shapes
 ▒
Read_7/ReadVariableOpReadVariableOp5read_7_disablecopyonread_batch_normalization_111_beta^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: љ
Read_8/DisableCopyOnReadDisableCopyOnRead<read_8_disablecopyonread_batch_normalization_111_moving_mean"/device:CPU:0*
_output_shapes
 И
Read_8/ReadVariableOpReadVariableOp<read_8_disablecopyonread_batch_normalization_111_moving_mean^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: ћ
Read_9/DisableCopyOnReadDisableCopyOnRead@read_9_disablecopyonread_batch_normalization_111_moving_variance"/device:CPU:0*
_output_shapes
 ╝
Read_9/ReadVariableOpReadVariableOp@read_9_disablecopyonread_batch_normalization_111_moving_variance^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: ђ
Read_10/DisableCopyOnReadDisableCopyOnRead+read_10_disablecopyonread_conv2d_110_kernel"/device:CPU:0*
_output_shapes
 х
Read_10/ReadVariableOpReadVariableOp+read_10_disablecopyonread_conv2d_110_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
: @~
Read_11/DisableCopyOnReadDisableCopyOnRead)read_11_disablecopyonread_conv2d_110_bias"/device:CPU:0*
_output_shapes
 Д
Read_11/ReadVariableOpReadVariableOp)read_11_disablecopyonread_conv2d_110_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@ї
Read_12/DisableCopyOnReadDisableCopyOnRead7read_12_disablecopyonread_batch_normalization_112_gamma"/device:CPU:0*
_output_shapes
 х
Read_12/ReadVariableOpReadVariableOp7read_12_disablecopyonread_batch_normalization_112_gamma^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@І
Read_13/DisableCopyOnReadDisableCopyOnRead6read_13_disablecopyonread_batch_normalization_112_beta"/device:CPU:0*
_output_shapes
 ┤
Read_13/ReadVariableOpReadVariableOp6read_13_disablecopyonread_batch_normalization_112_beta^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@њ
Read_14/DisableCopyOnReadDisableCopyOnRead=read_14_disablecopyonread_batch_normalization_112_moving_mean"/device:CPU:0*
_output_shapes
 ╗
Read_14/ReadVariableOpReadVariableOp=read_14_disablecopyonread_batch_normalization_112_moving_mean^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@ќ
Read_15/DisableCopyOnReadDisableCopyOnReadAread_15_disablecopyonread_batch_normalization_112_moving_variance"/device:CPU:0*
_output_shapes
 ┐
Read_15/ReadVariableOpReadVariableOpAread_15_disablecopyonread_batch_normalization_112_moving_variance^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@ђ
Read_16/DisableCopyOnReadDisableCopyOnRead+read_16_disablecopyonread_conv2d_111_kernel"/device:CPU:0*
_output_shapes
 х
Read_16/ReadVariableOpReadVariableOp+read_16_disablecopyonread_conv2d_111_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@~
Read_17/DisableCopyOnReadDisableCopyOnRead)read_17_disablecopyonread_conv2d_111_bias"/device:CPU:0*
_output_shapes
 Д
Read_17/ReadVariableOpReadVariableOp)read_17_disablecopyonread_conv2d_111_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_dense_68_kernel"/device:CPU:0*
_output_shapes
 г
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_dense_68_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	└@*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	└@f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	└@|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_dense_68_bias"/device:CPU:0*
_output_shapes
 Ц
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_dense_68_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_dense_69_kernel"/device:CPU:0*
_output_shapes
 Ф
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_dense_69_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:@|
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_dense_69_bias"/device:CPU:0*
_output_shapes
 Ц
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_dense_69_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_22/DisableCopyOnReadDisableCopyOnRead#read_22_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Ю
Read_22/ReadVariableOpReadVariableOp#read_22_disablecopyonread_iteration^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 А
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_learning_rate^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: Њ
Read_24/DisableCopyOnReadDisableCopyOnRead>read_24_disablecopyonread_adam_m_batch_normalization_110_gamma"/device:CPU:0*
_output_shapes
 ╝
Read_24/ReadVariableOpReadVariableOp>read_24_disablecopyonread_adam_m_batch_normalization_110_gamma^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:Њ
Read_25/DisableCopyOnReadDisableCopyOnRead>read_25_disablecopyonread_adam_v_batch_normalization_110_gamma"/device:CPU:0*
_output_shapes
 ╝
Read_25/ReadVariableOpReadVariableOp>read_25_disablecopyonread_adam_v_batch_normalization_110_gamma^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:њ
Read_26/DisableCopyOnReadDisableCopyOnRead=read_26_disablecopyonread_adam_m_batch_normalization_110_beta"/device:CPU:0*
_output_shapes
 ╗
Read_26/ReadVariableOpReadVariableOp=read_26_disablecopyonread_adam_m_batch_normalization_110_beta^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:њ
Read_27/DisableCopyOnReadDisableCopyOnRead=read_27_disablecopyonread_adam_v_batch_normalization_110_beta"/device:CPU:0*
_output_shapes
 ╗
Read_27/ReadVariableOpReadVariableOp=read_27_disablecopyonread_adam_v_batch_normalization_110_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:Є
Read_28/DisableCopyOnReadDisableCopyOnRead2read_28_disablecopyonread_adam_m_conv2d_109_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_28/ReadVariableOpReadVariableOp2read_28_disablecopyonread_adam_m_conv2d_109_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*&
_output_shapes
: Є
Read_29/DisableCopyOnReadDisableCopyOnRead2read_29_disablecopyonread_adam_v_conv2d_109_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_29/ReadVariableOpReadVariableOp2read_29_disablecopyonread_adam_v_conv2d_109_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
: Ё
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_conv2d_109_bias"/device:CPU:0*
_output_shapes
 «
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_conv2d_109_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: Ё
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_conv2d_109_bias"/device:CPU:0*
_output_shapes
 «
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_conv2d_109_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: Њ
Read_32/DisableCopyOnReadDisableCopyOnRead>read_32_disablecopyonread_adam_m_batch_normalization_111_gamma"/device:CPU:0*
_output_shapes
 ╝
Read_32/ReadVariableOpReadVariableOp>read_32_disablecopyonread_adam_m_batch_normalization_111_gamma^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: Њ
Read_33/DisableCopyOnReadDisableCopyOnRead>read_33_disablecopyonread_adam_v_batch_normalization_111_gamma"/device:CPU:0*
_output_shapes
 ╝
Read_33/ReadVariableOpReadVariableOp>read_33_disablecopyonread_adam_v_batch_normalization_111_gamma^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: њ
Read_34/DisableCopyOnReadDisableCopyOnRead=read_34_disablecopyonread_adam_m_batch_normalization_111_beta"/device:CPU:0*
_output_shapes
 ╗
Read_34/ReadVariableOpReadVariableOp=read_34_disablecopyonread_adam_m_batch_normalization_111_beta^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: њ
Read_35/DisableCopyOnReadDisableCopyOnRead=read_35_disablecopyonread_adam_v_batch_normalization_111_beta"/device:CPU:0*
_output_shapes
 ╗
Read_35/ReadVariableOpReadVariableOp=read_35_disablecopyonread_adam_v_batch_normalization_111_beta^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: Є
Read_36/DisableCopyOnReadDisableCopyOnRead2read_36_disablecopyonread_adam_m_conv2d_110_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_36/ReadVariableOpReadVariableOp2read_36_disablecopyonread_adam_m_conv2d_110_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Є
Read_37/DisableCopyOnReadDisableCopyOnRead2read_37_disablecopyonread_adam_v_conv2d_110_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_37/ReadVariableOpReadVariableOp2read_37_disablecopyonread_adam_v_conv2d_110_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Ё
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_conv2d_110_bias"/device:CPU:0*
_output_shapes
 «
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_conv2d_110_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ё
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_conv2d_110_bias"/device:CPU:0*
_output_shapes
 «
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_conv2d_110_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@Њ
Read_40/DisableCopyOnReadDisableCopyOnRead>read_40_disablecopyonread_adam_m_batch_normalization_112_gamma"/device:CPU:0*
_output_shapes
 ╝
Read_40/ReadVariableOpReadVariableOp>read_40_disablecopyonread_adam_m_batch_normalization_112_gamma^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@Њ
Read_41/DisableCopyOnReadDisableCopyOnRead>read_41_disablecopyonread_adam_v_batch_normalization_112_gamma"/device:CPU:0*
_output_shapes
 ╝
Read_41/ReadVariableOpReadVariableOp>read_41_disablecopyonread_adam_v_batch_normalization_112_gamma^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:@њ
Read_42/DisableCopyOnReadDisableCopyOnRead=read_42_disablecopyonread_adam_m_batch_normalization_112_beta"/device:CPU:0*
_output_shapes
 ╗
Read_42/ReadVariableOpReadVariableOp=read_42_disablecopyonread_adam_m_batch_normalization_112_beta^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:@њ
Read_43/DisableCopyOnReadDisableCopyOnRead=read_43_disablecopyonread_adam_v_batch_normalization_112_beta"/device:CPU:0*
_output_shapes
 ╗
Read_43/ReadVariableOpReadVariableOp=read_43_disablecopyonread_adam_v_batch_normalization_112_beta^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:@Є
Read_44/DisableCopyOnReadDisableCopyOnRead2read_44_disablecopyonread_adam_m_conv2d_111_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_44/ReadVariableOpReadVariableOp2read_44_disablecopyonread_adam_m_conv2d_111_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Є
Read_45/DisableCopyOnReadDisableCopyOnRead2read_45_disablecopyonread_adam_v_conv2d_111_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_45/ReadVariableOpReadVariableOp2read_45_disablecopyonread_adam_v_conv2d_111_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Ё
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_adam_m_conv2d_111_bias"/device:CPU:0*
_output_shapes
 «
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_adam_m_conv2d_111_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ё
Read_47/DisableCopyOnReadDisableCopyOnRead0read_47_disablecopyonread_adam_v_conv2d_111_bias"/device:CPU:0*
_output_shapes
 «
Read_47/ReadVariableOpReadVariableOp0read_47_disablecopyonread_adam_v_conv2d_111_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ё
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_dense_68_kernel"/device:CPU:0*
_output_shapes
 │
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_dense_68_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	└@*
dtype0p
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	└@f
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:	└@Ё
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_dense_68_kernel"/device:CPU:0*
_output_shapes
 │
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_dense_68_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	└@*
dtype0p
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	└@f
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:	└@Ѓ
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_dense_68_bias"/device:CPU:0*
_output_shapes
 г
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_dense_68_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ѓ
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_dense_68_bias"/device:CPU:0*
_output_shapes
 г
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_dense_68_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ё
Read_52/DisableCopyOnReadDisableCopyOnRead0read_52_disablecopyonread_adam_m_dense_69_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_52/ReadVariableOpReadVariableOp0read_52_disablecopyonread_adam_m_dense_69_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes

:@Ё
Read_53/DisableCopyOnReadDisableCopyOnRead0read_53_disablecopyonread_adam_v_dense_69_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_53/ReadVariableOpReadVariableOp0read_53_disablecopyonread_adam_v_dense_69_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes

:@Ѓ
Read_54/DisableCopyOnReadDisableCopyOnRead.read_54_disablecopyonread_adam_m_dense_69_bias"/device:CPU:0*
_output_shapes
 г
Read_54/ReadVariableOpReadVariableOp.read_54_disablecopyonread_adam_m_dense_69_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:Ѓ
Read_55/DisableCopyOnReadDisableCopyOnRead.read_55_disablecopyonread_adam_v_dense_69_bias"/device:CPU:0*
_output_shapes
 г
Read_55/ReadVariableOpReadVariableOp.read_55_disablecopyonread_adam_v_dense_69_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_56/DisableCopyOnReadDisableCopyOnRead!read_56_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Џ
Read_56/ReadVariableOpReadVariableOp!read_56_disablecopyonread_total_1^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_57/DisableCopyOnReadDisableCopyOnRead!read_57_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Џ
Read_57/ReadVariableOpReadVariableOp!read_57_disablecopyonread_count_1^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_58/DisableCopyOnReadDisableCopyOnReadread_58_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Ў
Read_58/ReadVariableOpReadVariableOpread_58_disablecopyonread_total^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_59/DisableCopyOnReadDisableCopyOnReadread_59_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ў
Read_59/ReadVariableOpReadVariableOpread_59_disablecopyonread_count^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: Д
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*л
valueкB├=B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*Ј
valueЁBѓ=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┴
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *K
dtypesA
?2=	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_120Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_121IdentityIdentity_120:output:0^NoOp*
T0*
_output_shapes
: І
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_121Identity_121:output:0*(
_construction_contextkEagerRuntime*Ј
_input_shapes~
|: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:==9

_output_shapes
: 

_user_specified_nameConst:%<!

_user_specified_namecount:%;!

_user_specified_nametotal:':#
!
_user_specified_name	count_1:'9#
!
_user_specified_name	total_1:480
.
_user_specified_nameAdam/v/dense_69/bias:470
.
_user_specified_nameAdam/m/dense_69/bias:662
0
_user_specified_nameAdam/v/dense_69/kernel:652
0
_user_specified_nameAdam/m/dense_69/kernel:440
.
_user_specified_nameAdam/v/dense_68/bias:430
.
_user_specified_nameAdam/m/dense_68/bias:622
0
_user_specified_nameAdam/v/dense_68/kernel:612
0
_user_specified_nameAdam/m/dense_68/kernel:602
0
_user_specified_nameAdam/v/conv2d_111/bias:6/2
0
_user_specified_nameAdam/m/conv2d_111/bias:8.4
2
_user_specified_nameAdam/v/conv2d_111/kernel:8-4
2
_user_specified_nameAdam/m/conv2d_111/kernel:C,?
=
_user_specified_name%#Adam/v/batch_normalization_112/beta:C+?
=
_user_specified_name%#Adam/m/batch_normalization_112/beta:D*@
>
_user_specified_name&$Adam/v/batch_normalization_112/gamma:D)@
>
_user_specified_name&$Adam/m/batch_normalization_112/gamma:6(2
0
_user_specified_nameAdam/v/conv2d_110/bias:6'2
0
_user_specified_nameAdam/m/conv2d_110/bias:8&4
2
_user_specified_nameAdam/v/conv2d_110/kernel:8%4
2
_user_specified_nameAdam/m/conv2d_110/kernel:C$?
=
_user_specified_name%#Adam/v/batch_normalization_111/beta:C#?
=
_user_specified_name%#Adam/m/batch_normalization_111/beta:D"@
>
_user_specified_name&$Adam/v/batch_normalization_111/gamma:D!@
>
_user_specified_name&$Adam/m/batch_normalization_111/gamma:6 2
0
_user_specified_nameAdam/v/conv2d_109/bias:62
0
_user_specified_nameAdam/m/conv2d_109/bias:84
2
_user_specified_nameAdam/v/conv2d_109/kernel:84
2
_user_specified_nameAdam/m/conv2d_109/kernel:C?
=
_user_specified_name%#Adam/v/batch_normalization_110/beta:C?
=
_user_specified_name%#Adam/m/batch_normalization_110/beta:D@
>
_user_specified_name&$Adam/v/batch_normalization_110/gamma:D@
>
_user_specified_name&$Adam/m/batch_normalization_110/gamma:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_69/bias:/+
)
_user_specified_namedense_69/kernel:-)
'
_user_specified_namedense_68/bias:/+
)
_user_specified_namedense_68/kernel:/+
)
_user_specified_nameconv2d_111/bias:1-
+
_user_specified_nameconv2d_111/kernel:GC
A
_user_specified_name)'batch_normalization_112/moving_variance:C?
=
_user_specified_name%#batch_normalization_112/moving_mean:<8
6
_user_specified_namebatch_normalization_112/beta:=9
7
_user_specified_namebatch_normalization_112/gamma:/+
)
_user_specified_nameconv2d_110/bias:1-
+
_user_specified_nameconv2d_110/kernel:G
C
A
_user_specified_name)'batch_normalization_111/moving_variance:C	?
=
_user_specified_name%#batch_normalization_111/moving_mean:<8
6
_user_specified_namebatch_normalization_111/beta:=9
7
_user_specified_namebatch_normalization_111/gamma:/+
)
_user_specified_nameconv2d_109/bias:1-
+
_user_specified_nameconv2d_109/kernel:GC
A
_user_specified_name)'batch_normalization_110/moving_variance:C?
=
_user_specified_name%#batch_normalization_110/moving_mean:<8
6
_user_specified_namebatch_normalization_110/beta:=9
7
_user_specified_namebatch_normalization_110/gamma:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
м
ъ
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_582921

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╩

ш
D__inference_dense_69_layer_call_and_return_conditional_losses_583223

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
│
G
+__inference_flatten_34_layer_call_fn_583869

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_34_layer_call_and_return_conditional_losses_583195a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
м
ъ
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583065

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ЫЅ
┬
!__inference__wrapped_model_582885!
batch_normalization_110_inputK
=sequential_34_batch_normalization_110_readvariableop_resource:M
?sequential_34_batch_normalization_110_readvariableop_1_resource:\
Nsequential_34_batch_normalization_110_fusedbatchnormv3_readvariableop_resource:^
Psequential_34_batch_normalization_110_fusedbatchnormv3_readvariableop_1_resource:Q
7sequential_34_conv2d_109_conv2d_readvariableop_resource: F
8sequential_34_conv2d_109_biasadd_readvariableop_resource: K
=sequential_34_batch_normalization_111_readvariableop_resource: M
?sequential_34_batch_normalization_111_readvariableop_1_resource: \
Nsequential_34_batch_normalization_111_fusedbatchnormv3_readvariableop_resource: ^
Psequential_34_batch_normalization_111_fusedbatchnormv3_readvariableop_1_resource: Q
7sequential_34_conv2d_110_conv2d_readvariableop_resource: @F
8sequential_34_conv2d_110_biasadd_readvariableop_resource:@K
=sequential_34_batch_normalization_112_readvariableop_resource:@M
?sequential_34_batch_normalization_112_readvariableop_1_resource:@\
Nsequential_34_batch_normalization_112_fusedbatchnormv3_readvariableop_resource:@^
Psequential_34_batch_normalization_112_fusedbatchnormv3_readvariableop_1_resource:@Q
7sequential_34_conv2d_111_conv2d_readvariableop_resource:@@F
8sequential_34_conv2d_111_biasadd_readvariableop_resource:@H
5sequential_34_dense_68_matmul_readvariableop_resource:	└@D
6sequential_34_dense_68_biasadd_readvariableop_resource:@G
5sequential_34_dense_69_matmul_readvariableop_resource:@D
6sequential_34_dense_69_biasadd_readvariableop_resource:
identityѕбEsequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOpбGsequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1б4sequential_34/batch_normalization_110/ReadVariableOpб6sequential_34/batch_normalization_110/ReadVariableOp_1бEsequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOpбGsequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1б4sequential_34/batch_normalization_111/ReadVariableOpб6sequential_34/batch_normalization_111/ReadVariableOp_1бEsequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOpбGsequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1б4sequential_34/batch_normalization_112/ReadVariableOpб6sequential_34/batch_normalization_112/ReadVariableOp_1б/sequential_34/conv2d_109/BiasAdd/ReadVariableOpб.sequential_34/conv2d_109/Conv2D/ReadVariableOpб/sequential_34/conv2d_110/BiasAdd/ReadVariableOpб.sequential_34/conv2d_110/Conv2D/ReadVariableOpб/sequential_34/conv2d_111/BiasAdd/ReadVariableOpб.sequential_34/conv2d_111/Conv2D/ReadVariableOpб-sequential_34/dense_68/BiasAdd/ReadVariableOpб,sequential_34/dense_68/MatMul/ReadVariableOpб-sequential_34/dense_69/BiasAdd/ReadVariableOpб,sequential_34/dense_69/MatMul/ReadVariableOpџ
*sequential_34/batch_normalization_110/CastCastbatch_normalization_110_input*

DstT0*

SrcT0*/
_output_shapes
:         «
4sequential_34/batch_normalization_110/ReadVariableOpReadVariableOp=sequential_34_batch_normalization_110_readvariableop_resource*
_output_shapes
:*
dtype0▓
6sequential_34/batch_normalization_110/ReadVariableOp_1ReadVariableOp?sequential_34_batch_normalization_110_readvariableop_1_resource*
_output_shapes
:*
dtype0л
Esequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_34_batch_normalization_110_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0н
Gsequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_34_batch_normalization_110_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ю
6sequential_34/batch_normalization_110/FusedBatchNormV3FusedBatchNormV3.sequential_34/batch_normalization_110/Cast:y:0<sequential_34/batch_normalization_110/ReadVariableOp:value:0>sequential_34/batch_normalization_110/ReadVariableOp_1:value:0Msequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp:value:0Osequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
is_training( «
.sequential_34/conv2d_109/Conv2D/ReadVariableOpReadVariableOp7sequential_34_conv2d_109_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ђ
sequential_34/conv2d_109/Conv2DConv2D:sequential_34/batch_normalization_110/FusedBatchNormV3:y:06sequential_34/conv2d_109/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
ц
/sequential_34/conv2d_109/BiasAdd/ReadVariableOpReadVariableOp8sequential_34_conv2d_109_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╚
 sequential_34/conv2d_109/BiasAddBiasAdd(sequential_34/conv2d_109/Conv2D:output:07sequential_34/conv2d_109/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          і
sequential_34/conv2d_109/ReluRelu)sequential_34/conv2d_109/BiasAdd:output:0*
T0*/
_output_shapes
:          ╦
&sequential_34/max_pooling2d_77/MaxPoolMaxPool+sequential_34/conv2d_109/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
«
4sequential_34/batch_normalization_111/ReadVariableOpReadVariableOp=sequential_34_batch_normalization_111_readvariableop_resource*
_output_shapes
: *
dtype0▓
6sequential_34/batch_normalization_111/ReadVariableOp_1ReadVariableOp?sequential_34_batch_normalization_111_readvariableop_1_resource*
_output_shapes
: *
dtype0л
Esequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_34_batch_normalization_111_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0н
Gsequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_34_batch_normalization_111_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ю
6sequential_34/batch_normalization_111/FusedBatchNormV3FusedBatchNormV3/sequential_34/max_pooling2d_77/MaxPool:output:0<sequential_34/batch_normalization_111/ReadVariableOp:value:0>sequential_34/batch_normalization_111/ReadVariableOp_1:value:0Msequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp:value:0Osequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( «
.sequential_34/conv2d_110/Conv2D/ReadVariableOpReadVariableOp7sequential_34_conv2d_110_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ђ
sequential_34/conv2d_110/Conv2DConv2D:sequential_34/batch_normalization_111/FusedBatchNormV3:y:06sequential_34/conv2d_110/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
ц
/sequential_34/conv2d_110/BiasAdd/ReadVariableOpReadVariableOp8sequential_34_conv2d_110_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╚
 sequential_34/conv2d_110/BiasAddBiasAdd(sequential_34/conv2d_110/Conv2D:output:07sequential_34/conv2d_110/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @і
sequential_34/conv2d_110/ReluRelu)sequential_34/conv2d_110/BiasAdd:output:0*
T0*/
_output_shapes
:         @╦
&sequential_34/max_pooling2d_78/MaxPoolMaxPool+sequential_34/conv2d_110/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
«
4sequential_34/batch_normalization_112/ReadVariableOpReadVariableOp=sequential_34_batch_normalization_112_readvariableop_resource*
_output_shapes
:@*
dtype0▓
6sequential_34/batch_normalization_112/ReadVariableOp_1ReadVariableOp?sequential_34_batch_normalization_112_readvariableop_1_resource*
_output_shapes
:@*
dtype0л
Esequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_34_batch_normalization_112_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0н
Gsequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_34_batch_normalization_112_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ю
6sequential_34/batch_normalization_112/FusedBatchNormV3FusedBatchNormV3/sequential_34/max_pooling2d_78/MaxPool:output:0<sequential_34/batch_normalization_112/ReadVariableOp:value:0>sequential_34/batch_normalization_112/ReadVariableOp_1:value:0Msequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp:value:0Osequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( «
.sequential_34/conv2d_111/Conv2D/ReadVariableOpReadVariableOp7sequential_34_conv2d_111_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ђ
sequential_34/conv2d_111/Conv2DConv2D:sequential_34/batch_normalization_112/FusedBatchNormV3:y:06sequential_34/conv2d_111/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
ц
/sequential_34/conv2d_111/BiasAdd/ReadVariableOpReadVariableOp8sequential_34_conv2d_111_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╚
 sequential_34/conv2d_111/BiasAddBiasAdd(sequential_34/conv2d_111/Conv2D:output:07sequential_34/conv2d_111/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @і
sequential_34/conv2d_111/ReluRelu)sequential_34/conv2d_111/BiasAdd:output:0*
T0*/
_output_shapes
:         @o
sequential_34/flatten_34/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  ┤
 sequential_34/flatten_34/ReshapeReshape+sequential_34/conv2d_111/Relu:activations:0'sequential_34/flatten_34/Const:output:0*
T0*(
_output_shapes
:         └Б
,sequential_34/dense_68/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_68_matmul_readvariableop_resource*
_output_shapes
:	└@*
dtype0║
sequential_34/dense_68/MatMulMatMul)sequential_34/flatten_34/Reshape:output:04sequential_34/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
-sequential_34/dense_68/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_68_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╗
sequential_34/dense_68/BiasAddBiasAdd'sequential_34/dense_68/MatMul:product:05sequential_34/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @~
sequential_34/dense_68/ReluRelu'sequential_34/dense_68/BiasAdd:output:0*
T0*'
_output_shapes
:         @б
,sequential_34/dense_69/MatMul/ReadVariableOpReadVariableOp5sequential_34_dense_69_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0║
sequential_34/dense_69/MatMulMatMul)sequential_34/dense_68/Relu:activations:04sequential_34/dense_69/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_34/dense_69/BiasAdd/ReadVariableOpReadVariableOp6sequential_34_dense_69_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_34/dense_69/BiasAddBiasAdd'sequential_34/dense_69/MatMul:product:05sequential_34/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
sequential_34/dense_69/SigmoidSigmoid'sequential_34/dense_69/BiasAdd:output:0*
T0*'
_output_shapes
:         q
IdentityIdentity"sequential_34/dense_69/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         Ј

NoOpNoOpF^sequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOpH^sequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_15^sequential_34/batch_normalization_110/ReadVariableOp7^sequential_34/batch_normalization_110/ReadVariableOp_1F^sequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOpH^sequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_15^sequential_34/batch_normalization_111/ReadVariableOp7^sequential_34/batch_normalization_111/ReadVariableOp_1F^sequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOpH^sequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_15^sequential_34/batch_normalization_112/ReadVariableOp7^sequential_34/batch_normalization_112/ReadVariableOp_10^sequential_34/conv2d_109/BiasAdd/ReadVariableOp/^sequential_34/conv2d_109/Conv2D/ReadVariableOp0^sequential_34/conv2d_110/BiasAdd/ReadVariableOp/^sequential_34/conv2d_110/Conv2D/ReadVariableOp0^sequential_34/conv2d_111/BiasAdd/ReadVariableOp/^sequential_34/conv2d_111/Conv2D/ReadVariableOp.^sequential_34/dense_68/BiasAdd/ReadVariableOp-^sequential_34/dense_68/MatMul/ReadVariableOp.^sequential_34/dense_69/BiasAdd/ReadVariableOp-^sequential_34/dense_69/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         : : : : : : : : : : : : : : : : : : : : : : 2њ
Gsequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_1Gsequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp_12ј
Esequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOpEsequential_34/batch_normalization_110/FusedBatchNormV3/ReadVariableOp2p
6sequential_34/batch_normalization_110/ReadVariableOp_16sequential_34/batch_normalization_110/ReadVariableOp_12l
4sequential_34/batch_normalization_110/ReadVariableOp4sequential_34/batch_normalization_110/ReadVariableOp2њ
Gsequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_1Gsequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp_12ј
Esequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOpEsequential_34/batch_normalization_111/FusedBatchNormV3/ReadVariableOp2p
6sequential_34/batch_normalization_111/ReadVariableOp_16sequential_34/batch_normalization_111/ReadVariableOp_12l
4sequential_34/batch_normalization_111/ReadVariableOp4sequential_34/batch_normalization_111/ReadVariableOp2њ
Gsequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_1Gsequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp_12ј
Esequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOpEsequential_34/batch_normalization_112/FusedBatchNormV3/ReadVariableOp2p
6sequential_34/batch_normalization_112/ReadVariableOp_16sequential_34/batch_normalization_112/ReadVariableOp_12l
4sequential_34/batch_normalization_112/ReadVariableOp4sequential_34/batch_normalization_112/ReadVariableOp2b
/sequential_34/conv2d_109/BiasAdd/ReadVariableOp/sequential_34/conv2d_109/BiasAdd/ReadVariableOp2`
.sequential_34/conv2d_109/Conv2D/ReadVariableOp.sequential_34/conv2d_109/Conv2D/ReadVariableOp2b
/sequential_34/conv2d_110/BiasAdd/ReadVariableOp/sequential_34/conv2d_110/BiasAdd/ReadVariableOp2`
.sequential_34/conv2d_110/Conv2D/ReadVariableOp.sequential_34/conv2d_110/Conv2D/ReadVariableOp2b
/sequential_34/conv2d_111/BiasAdd/ReadVariableOp/sequential_34/conv2d_111/BiasAdd/ReadVariableOp2`
.sequential_34/conv2d_111/Conv2D/ReadVariableOp.sequential_34/conv2d_111/Conv2D/ReadVariableOp2^
-sequential_34/dense_68/BiasAdd/ReadVariableOp-sequential_34/dense_68/BiasAdd/ReadVariableOp2\
,sequential_34/dense_68/MatMul/ReadVariableOp,sequential_34/dense_68/MatMul/ReadVariableOp2^
-sequential_34/dense_69/BiasAdd/ReadVariableOp-sequential_34/dense_69/BiasAdd/ReadVariableOp2\
,sequential_34/dense_69/MatMul/ReadVariableOp,sequential_34/dense_69/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:n j
/
_output_shapes
:         
7
_user_specified_namebatch_normalization_110_input
Х
 
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583864

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
¤

Ш
D__inference_dense_68_layer_call_and_return_conditional_losses_583895

inputs1
matmul_readvariableop_resource:	└@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ћ
h
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583024

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
м
ъ
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583844

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╚	
М
8__inference_batch_normalization_110_layer_call_fn_583586

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583251w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583582:&"
 
_user_specified_name583580:&"
 
_user_specified_name583578:&"
 
_user_specified_name583576:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
џ
а
+__inference_conv2d_110_layer_call_fn_583761

inputs!
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583158w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583757:&"
 
_user_specified_name583755:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╩

ш
D__inference_dense_69_layer_call_and_return_conditional_losses_583915

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╚
b
F__inference_flatten_34_layer_call_and_return_conditional_losses_583195

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Х
 
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583772

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╚
b
F__inference_flatten_34_layer_call_and_return_conditional_losses_583875

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Ц
┬
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583112

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0к
FusedBatchNormV3FusedBatchNormV3Cast:y:0ReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         ░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ћ
h
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583782

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ј

М
8__inference_batch_normalization_111_layer_call_fn_583703

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_582975Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583699:&"
 
_user_specified_name583697:&"
 
_user_specified_name583695:&"
 
_user_specified_name583693:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
џ
а
+__inference_conv2d_109_layer_call_fn_583669

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583132w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583665:&"
 
_user_specified_name583663:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
м
ъ
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583622

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ї
┬
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_582975

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ј

М
8__inference_batch_normalization_110_layer_call_fn_583547

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_582903Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583543:&"
 
_user_specified_name583541:&"
 
_user_specified_name583539:&"
 
_user_specified_name583537:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ї
┬
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583047

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╦B
р

I__inference_sequential_34_layer_call_and_return_conditional_losses_583230!
batch_normalization_110_input,
batch_normalization_110_583113:,
batch_normalization_110_583115:,
batch_normalization_110_583117:,
batch_normalization_110_583119:+
conv2d_109_583133: 
conv2d_109_583135: ,
batch_normalization_111_583139: ,
batch_normalization_111_583141: ,
batch_normalization_111_583143: ,
batch_normalization_111_583145: +
conv2d_110_583159: @
conv2d_110_583161:@,
batch_normalization_112_583165:@,
batch_normalization_112_583167:@,
batch_normalization_112_583169:@,
batch_normalization_112_583171:@+
conv2d_111_583185:@@
conv2d_111_583187:@"
dense_68_583208:	└@
dense_68_583210:@!
dense_69_583224:@
dense_69_583226:
identityѕб/batch_normalization_110/StatefulPartitionedCallб/batch_normalization_111/StatefulPartitionedCallб/batch_normalization_112/StatefulPartitionedCallб"conv2d_109/StatefulPartitionedCallб"conv2d_110/StatefulPartitionedCallб"conv2d_111/StatefulPartitionedCallб dense_68/StatefulPartitionedCallб dense_69/StatefulPartitionedCallЇ
/batch_normalization_110/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_110_inputbatch_normalization_110_583113batch_normalization_110_583115batch_normalization_110_583117batch_normalization_110_583119*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583112▓
"conv2d_109/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_110/StatefulPartitionedCall:output:0conv2d_109_583133conv2d_109_583135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583132ш
 max_pooling2d_77/PartitionedCallPartitionedCall+conv2d_109/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_582952Ў
/batch_normalization_111/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_77/PartitionedCall:output:0batch_normalization_111_583139batch_normalization_111_583141batch_normalization_111_583143batch_normalization_111_583145*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_582975▓
"conv2d_110/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_111/StatefulPartitionedCall:output:0conv2d_110_583159conv2d_110_583161*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583158ш
 max_pooling2d_78/PartitionedCallPartitionedCall+conv2d_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583024Ў
/batch_normalization_112/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_78/PartitionedCall:output:0batch_normalization_112_583165batch_normalization_112_583167batch_normalization_112_583169batch_normalization_112_583171*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583047▓
"conv2d_111/StatefulPartitionedCallStatefulPartitionedCall8batch_normalization_112/StatefulPartitionedCall:output:0conv2d_111_583185conv2d_111_583187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583184Р
flatten_34/PartitionedCallPartitionedCall+conv2d_111/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_34_layer_call_and_return_conditional_losses_583195Ї
 dense_68/StatefulPartitionedCallStatefulPartitionedCall#flatten_34/PartitionedCall:output:0dense_68_583208dense_68_583210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_583207Њ
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_583224dense_69_583226*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_69_layer_call_and_return_conditional_losses_583223x
IdentityIdentity)dense_69/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ь
NoOpNoOp0^batch_normalization_110/StatefulPartitionedCall0^batch_normalization_111/StatefulPartitionedCall0^batch_normalization_112/StatefulPartitionedCall#^conv2d_109/StatefulPartitionedCall#^conv2d_110/StatefulPartitionedCall#^conv2d_111/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall!^dense_69/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:         : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_normalization_110/StatefulPartitionedCall/batch_normalization_110/StatefulPartitionedCall2b
/batch_normalization_111/StatefulPartitionedCall/batch_normalization_111/StatefulPartitionedCall2b
/batch_normalization_112/StatefulPartitionedCall/batch_normalization_112/StatefulPartitionedCall2H
"conv2d_109/StatefulPartitionedCall"conv2d_109/StatefulPartitionedCall2H
"conv2d_110/StatefulPartitionedCall"conv2d_110/StatefulPartitionedCall2H
"conv2d_111/StatefulPartitionedCall"conv2d_111/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall:&"
 
_user_specified_name583226:&"
 
_user_specified_name583224:&"
 
_user_specified_name583210:&"
 
_user_specified_name583208:&"
 
_user_specified_name583187:&"
 
_user_specified_name583185:&"
 
_user_specified_name583171:&"
 
_user_specified_name583169:&"
 
_user_specified_name583167:&"
 
_user_specified_name583165:&"
 
_user_specified_name583161:&"
 
_user_specified_name583159:&
"
 
_user_specified_name583145:&	"
 
_user_specified_name583143:&"
 
_user_specified_name583141:&"
 
_user_specified_name583139:&"
 
_user_specified_name583135:&"
 
_user_specified_name583133:&"
 
_user_specified_name583119:&"
 
_user_specified_name583117:&"
 
_user_specified_name583115:&"
 
_user_specified_name583113:n j
/
_output_shapes
:         
7
_user_specified_namebatch_normalization_110_input
ы
Ќ
)__inference_dense_68_layer_call_fn_583884

inputs
unknown:	└@
	unknown_0:@
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_583207o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583880:&"
 
_user_specified_name583878:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ћ
h
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_583690

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к	
М
8__inference_batch_normalization_110_layer_call_fn_583573

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583112w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583569:&"
 
_user_specified_name583567:&"
 
_user_specified_name583565:&"
 
_user_specified_name583563:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
¤

Ш
D__inference_dense_68_layer_call_and_return_conditional_losses_583207

inputs1
matmul_readvariableop_resource:	└@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ї
┬
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583826

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Х
 
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583184

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Х
 
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583158

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Љ

М
8__inference_batch_normalization_112_layer_call_fn_583808

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583065Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583804:&"
 
_user_specified_name583802:&"
 
_user_specified_name583800:&"
 
_user_specified_name583798:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
в
ъ
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583251

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0И
FusedBatchNormV3FusedBatchNormV3Cast:y:0ReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
в
ъ
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583660

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1]
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:         b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0И
FusedBatchNormV3FusedBatchNormV3Cast:y:0ReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         :::::*
epsilon%oЃ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         ї
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ћ
h
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_582952

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Љ

М
8__inference_batch_normalization_110_layer_call_fn_583560

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *\
fWRU
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_582921Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583556:&"
 
_user_specified_name583554:&"
 
_user_specified_name583552:&"
 
_user_specified_name583550:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
џ
а
+__inference_conv2d_111_layer_call_fn_583853

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583184w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name583849:&"
 
_user_specified_name583847:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs"ьL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▀
serving_default╦
o
batch_normalization_110_inputN
/serving_default_batch_normalization_110_input:0         <
dense_690
StatefulPartitionedCall:0         tensorflow/serving/predict:ли
к
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
П
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
Ц
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
Ж
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5axis
	6gamma
7beta
8moving_mean
9moving_variance"
_tf_keras_layer
П
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op"
_tf_keras_layer
Ц
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
Ж
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance"
_tf_keras_layer
П
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias
 \_jit_compiled_convolution_op"
_tf_keras_layer
Ц
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias"
_tf_keras_layer
╗
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias"
_tf_keras_layer
к
0
1
2
3
&4
'5
66
77
88
99
@10
A11
P12
Q13
R14
S15
Z16
[17
i18
j19
q20
r21"
trackable_list_wrapper
ќ
0
1
&2
'3
64
75
@6
A7
P8
Q9
Z10
[11
i12
j13
q14
r15"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¤
xtrace_0
ytrace_12ў
.__inference_sequential_34_layer_call_fn_583357
.__inference_sequential_34_layer_call_fn_583406х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zxtrace_0zytrace_1
Ё
ztrace_0
{trace_12╬
I__inference_sequential_34_layer_call_and_return_conditional_losses_583230
I__inference_sequential_34_layer_call_and_return_conditional_losses_583308х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zztrace_0z{trace_1
РB▀
!__inference__wrapped_model_582885batch_normalization_110_input"ў
Љ▓Ї
FullArgSpec
argsџ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ъ
|
_variables
}_iterations
~_learning_rate
_index_dict
ђ
_momentums
Ђ_velocities
ѓ_update_step_xla"
experimentalOptimizer
-
Ѓserving_default"
signature_map
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Њ
Ѕtrace_0
іtrace_1
Іtrace_2
їtrace_32а
8__inference_batch_normalization_110_layer_call_fn_583547
8__inference_batch_normalization_110_layer_call_fn_583560
8__inference_batch_normalization_110_layer_call_fn_583573
8__inference_batch_normalization_110_layer_call_fn_583586х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЅtrace_0zіtrace_1zІtrace_2zїtrace_3
 
Їtrace_0
јtrace_1
Јtrace_2
љtrace_32ї
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583604
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583622
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583641
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583660х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0zјtrace_1zЈtrace_2zљtrace_3
 "
trackable_list_wrapper
+:)2batch_normalization_110/gamma
*:(2batch_normalization_110/beta
3:1 (2#batch_normalization_110/moving_mean
7:5 (2'batch_normalization_110/moving_variance
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Љnon_trainable_variables
њlayers
Њmetrics
 ћlayer_regularization_losses
Ћlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
у
ќtrace_02╚
+__inference_conv2d_109_layer_call_fn_583669ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zќtrace_0
ѓ
Ќtrace_02с
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583680ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЌtrace_0
+:) 2conv2d_109/kernel
: 2conv2d_109/bias
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ь
Юtrace_02╬
1__inference_max_pooling2d_77_layer_call_fn_583685ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЮtrace_0
ѕ
ъtrace_02ж
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_583690ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zъtrace_0
<
60
71
82
93"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ъnon_trainable_variables
аlayers
Аmetrics
 бlayer_regularization_losses
Бlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
у
цtrace_0
Цtrace_12г
8__inference_batch_normalization_111_layer_call_fn_583703
8__inference_batch_normalization_111_layer_call_fn_583716х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zцtrace_0zЦtrace_1
Ю
дtrace_0
Дtrace_12Р
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583734
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583752х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zдtrace_0zДtrace_1
 "
trackable_list_wrapper
+:) 2batch_normalization_111/gamma
*:( 2batch_normalization_111/beta
3:1  (2#batch_normalization_111/moving_mean
7:5  (2'batch_normalization_111/moving_variance
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
у
Гtrace_02╚
+__inference_conv2d_110_layer_call_fn_583761ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zГtrace_0
ѓ
«trace_02с
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583772ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z«trace_0
+:) @2conv2d_110/kernel
:@2conv2d_110/bias
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
»non_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
ь
┤trace_02╬
1__inference_max_pooling2d_78_layer_call_fn_583777ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┤trace_0
ѕ
хtrace_02ж
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583782ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zхtrace_0
<
P0
Q1
R2
S3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
у
╗trace_0
╝trace_12г
8__inference_batch_normalization_112_layer_call_fn_583795
8__inference_batch_normalization_112_layer_call_fn_583808х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╗trace_0z╝trace_1
Ю
йtrace_0
Йtrace_12Р
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583826
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583844х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zйtrace_0zЙtrace_1
 "
trackable_list_wrapper
+:)@2batch_normalization_112/gamma
*:(@2batch_normalization_112/beta
3:1@ (2#batch_normalization_112/moving_mean
7:5@ (2'batch_normalization_112/moving_variance
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
у
─trace_02╚
+__inference_conv2d_111_layer_call_fn_583853ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z─trace_0
ѓ
┼trace_02с
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583864ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┼trace_0
+:)@@2conv2d_111/kernel
:@2conv2d_111/bias
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
кnon_trainable_variables
Кlayers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
у
╦trace_02╚
+__inference_flatten_34_layer_call_fn_583869ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╦trace_0
ѓ
╠trace_02с
F__inference_flatten_34_layer_call_and_return_conditional_losses_583875ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╠trace_0
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
═non_trainable_variables
╬layers
¤metrics
 лlayer_regularization_losses
Лlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
т
мtrace_02к
)__inference_dense_68_layer_call_fn_583884ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zмtrace_0
ђ
Мtrace_02р
D__inference_dense_68_layer_call_and_return_conditional_losses_583895ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zМtrace_0
": 	└@2dense_68/kernel
:@2dense_68/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
нnon_trainable_variables
Нlayers
оmetrics
 Оlayer_regularization_losses
пlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
т
┘trace_02к
)__inference_dense_69_layer_call_fn_583904ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┘trace_0
ђ
┌trace_02р
D__inference_dense_69_layer_call_and_return_conditional_losses_583915ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┌trace_0
!:@2dense_69/kernel
:2dense_69/bias
J
0
1
82
93
R4
S5"
trackable_list_wrapper
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
10"
trackable_list_wrapper
0
█0
▄1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЃBђ
.__inference_sequential_34_layer_call_fn_583357batch_normalization_110_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЃBђ
.__inference_sequential_34_layer_call_fn_583406batch_normalization_110_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъBЏ
I__inference_sequential_34_layer_call_and_return_conditional_losses_583230batch_normalization_110_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъBЏ
I__inference_sequential_34_layer_call_and_return_conditional_losses_583308batch_normalization_110_input"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Й
}0
П1
я2
▀3
Я4
р5
Р6
с7
С8
т9
Т10
у11
У12
ж13
Ж14
в15
В16
ь17
Ь18
№19
­20
ы21
Ы22
з23
З24
ш25
Ш26
э27
Э28
щ29
Щ30
ч31
Ч32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
д
П0
▀1
р2
с3
т4
у5
ж6
в7
ь8
№9
ы10
з11
ш12
э13
щ14
ч15"
trackable_list_wrapper
д
я0
Я1
Р2
С3
Т4
У5
Ж6
В7
Ь8
­9
Ы10
З11
Ш12
Э13
Щ14
Ч15"
trackable_list_wrapper
х2▓»
д▓б
FullArgSpec*
args"џ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
ЧBщ
$__inference_signature_wrapper_583534batch_normalization_110_input"»
е▓ц
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 2

kwonlyargs$џ!
jbatch_normalization_110_input
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ШBз
8__inference_batch_normalization_110_layer_call_fn_583547inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
8__inference_batch_normalization_110_layer_call_fn_583560inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
8__inference_batch_normalization_110_layer_call_fn_583573inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
8__inference_batch_normalization_110_layer_call_fn_583586inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583604inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583622inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583641inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583660inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
НBм
+__inference_conv2d_109_layer_call_fn_583669inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583680inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
█Bп
1__inference_max_pooling2d_77_layer_call_fn_583685inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_583690inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ШBз
8__inference_batch_normalization_111_layer_call_fn_583703inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
8__inference_batch_normalization_111_layer_call_fn_583716inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583734inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583752inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
НBм
+__inference_conv2d_110_layer_call_fn_583761inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583772inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
█Bп
1__inference_max_pooling2d_78_layer_call_fn_583777inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583782inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ШBз
8__inference_batch_normalization_112_layer_call_fn_583795inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
8__inference_batch_normalization_112_layer_call_fn_583808inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583826inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583844inputs"г
Ц▓А
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
НBм
+__inference_conv2d_111_layer_call_fn_583853inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583864inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
НBм
+__inference_flatten_34_layer_call_fn_583869inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
F__inference_flatten_34_layer_call_and_return_conditional_losses_583875inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
МBл
)__inference_dense_68_layer_call_fn_583884inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЬBв
D__inference_dense_68_layer_call_and_return_conditional_losses_583895inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
МBл
)__inference_dense_69_layer_call_fn_583904inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЬBв
D__inference_dense_69_layer_call_and_return_conditional_losses_583915inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
§	variables
■	keras_api

 total

ђcount"
_tf_keras_metric
c
Ђ	variables
ѓ	keras_api

Ѓtotal

ёcount
Ё
_fn_kwargs"
_tf_keras_metric
0:.2$Adam/m/batch_normalization_110/gamma
0:.2$Adam/v/batch_normalization_110/gamma
/:-2#Adam/m/batch_normalization_110/beta
/:-2#Adam/v/batch_normalization_110/beta
0:. 2Adam/m/conv2d_109/kernel
0:. 2Adam/v/conv2d_109/kernel
":  2Adam/m/conv2d_109/bias
":  2Adam/v/conv2d_109/bias
0:. 2$Adam/m/batch_normalization_111/gamma
0:. 2$Adam/v/batch_normalization_111/gamma
/:- 2#Adam/m/batch_normalization_111/beta
/:- 2#Adam/v/batch_normalization_111/beta
0:. @2Adam/m/conv2d_110/kernel
0:. @2Adam/v/conv2d_110/kernel
": @2Adam/m/conv2d_110/bias
": @2Adam/v/conv2d_110/bias
0:.@2$Adam/m/batch_normalization_112/gamma
0:.@2$Adam/v/batch_normalization_112/gamma
/:-@2#Adam/m/batch_normalization_112/beta
/:-@2#Adam/v/batch_normalization_112/beta
0:.@@2Adam/m/conv2d_111/kernel
0:.@@2Adam/v/conv2d_111/kernel
": @2Adam/m/conv2d_111/bias
": @2Adam/v/conv2d_111/bias
':%	└@2Adam/m/dense_68/kernel
':%	└@2Adam/v/dense_68/kernel
 :@2Adam/m/dense_68/bias
 :@2Adam/v/dense_68/bias
&:$@2Adam/m/dense_69/kernel
&:$@2Adam/v/dense_69/kernel
 :2Adam/m/dense_69/bias
 :2Adam/v/dense_69/bias
0
 0
ђ1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
:  (2total
:  (2count
0
Ѓ0
ё1"
trackable_list_wrapper
.
Ђ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper├
!__inference__wrapped_model_582885Ю&'6789@APQRSZ[ijqrNбK
DбA
?і<
batch_normalization_110_input         
ф "3ф0
.
dense_69"і
dense_69         щ
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583604АQбN
GбD
:і7
inputs+                           
p

 
ф "FбC
<і9
tensor_0+                           
џ щ
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583622АQбN
GбD
:і7
inputs+                           
p 

 
ф "FбC
<і9
tensor_0+                           
џ н
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583641}?б<
5б2
(і%
inputs         
p

 
ф "4б1
*і'
tensor_0         
џ н
S__inference_batch_normalization_110_layer_call_and_return_conditional_losses_583660}?б<
5б2
(і%
inputs         
p 

 
ф "4б1
*і'
tensor_0         
џ М
8__inference_batch_normalization_110_layer_call_fn_583547ќQбN
GбD
:і7
inputs+                           
p

 
ф ";і8
unknown+                           М
8__inference_batch_normalization_110_layer_call_fn_583560ќQбN
GбD
:і7
inputs+                           
p 

 
ф ";і8
unknown+                           «
8__inference_batch_normalization_110_layer_call_fn_583573r?б<
5б2
(і%
inputs         
p

 
ф ")і&
unknown         «
8__inference_batch_normalization_110_layer_call_fn_583586r?б<
5б2
(і%
inputs         
p 

 
ф ")і&
unknown         щ
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583734А6789QбN
GбD
:і7
inputs+                            
p

 
ф "FбC
<і9
tensor_0+                            
џ щ
S__inference_batch_normalization_111_layer_call_and_return_conditional_losses_583752А6789QбN
GбD
:і7
inputs+                            
p 

 
ф "FбC
<і9
tensor_0+                            
џ М
8__inference_batch_normalization_111_layer_call_fn_583703ќ6789QбN
GбD
:і7
inputs+                            
p

 
ф ";і8
unknown+                            М
8__inference_batch_normalization_111_layer_call_fn_583716ќ6789QбN
GбD
:і7
inputs+                            
p 

 
ф ";і8
unknown+                            щ
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583826АPQRSQбN
GбD
:і7
inputs+                           @
p

 
ф "FбC
<і9
tensor_0+                           @
џ щ
S__inference_batch_normalization_112_layer_call_and_return_conditional_losses_583844АPQRSQбN
GбD
:і7
inputs+                           @
p 

 
ф "FбC
<і9
tensor_0+                           @
џ М
8__inference_batch_normalization_112_layer_call_fn_583795ќPQRSQбN
GбD
:і7
inputs+                           @
p

 
ф ";і8
unknown+                           @М
8__inference_batch_normalization_112_layer_call_fn_583808ќPQRSQбN
GбD
:і7
inputs+                           @
p 

 
ф ";і8
unknown+                           @й
F__inference_conv2d_109_layer_call_and_return_conditional_losses_583680s&'7б4
-б*
(і%
inputs         
ф "4б1
*і'
tensor_0          
џ Ќ
+__inference_conv2d_109_layer_call_fn_583669h&'7б4
-б*
(і%
inputs         
ф ")і&
unknown          й
F__inference_conv2d_110_layer_call_and_return_conditional_losses_583772s@A7б4
-б*
(і%
inputs          
ф "4б1
*і'
tensor_0         @
џ Ќ
+__inference_conv2d_110_layer_call_fn_583761h@A7б4
-б*
(і%
inputs          
ф ")і&
unknown         @й
F__inference_conv2d_111_layer_call_and_return_conditional_losses_583864sZ[7б4
-б*
(і%
inputs         @
ф "4б1
*і'
tensor_0         @
џ Ќ
+__inference_conv2d_111_layer_call_fn_583853hZ[7б4
-б*
(і%
inputs         @
ф ")і&
unknown         @г
D__inference_dense_68_layer_call_and_return_conditional_losses_583895dij0б-
&б#
!і
inputs         └
ф ",б)
"і
tensor_0         @
џ є
)__inference_dense_68_layer_call_fn_583884Yij0б-
&б#
!і
inputs         └
ф "!і
unknown         @Ф
D__inference_dense_69_layer_call_and_return_conditional_losses_583915cqr/б,
%б"
 і
inputs         @
ф ",б)
"і
tensor_0         
џ Ё
)__inference_dense_69_layer_call_fn_583904Xqr/б,
%б"
 і
inputs         @
ф "!і
unknown         ▓
F__inference_flatten_34_layer_call_and_return_conditional_losses_583875h7б4
-б*
(і%
inputs         @
ф "-б*
#і 
tensor_0         └
џ ї
+__inference_flatten_34_layer_call_fn_583869]7б4
-б*
(і%
inputs         @
ф ""і
unknown         └Ш
L__inference_max_pooling2d_77_layer_call_and_return_conditional_losses_583690ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ л
1__inference_max_pooling2d_77_layer_call_fn_583685џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    Ш
L__inference_max_pooling2d_78_layer_call_and_return_conditional_losses_583782ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ л
1__inference_max_pooling2d_78_layer_call_fn_583777џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    В
I__inference_sequential_34_layer_call_and_return_conditional_losses_583230ъ&'6789@APQRSZ[ijqrVбS
LбI
?і<
batch_normalization_110_input         
p

 
ф ",б)
"і
tensor_0         
џ В
I__inference_sequential_34_layer_call_and_return_conditional_losses_583308ъ&'6789@APQRSZ[ijqrVбS
LбI
?і<
batch_normalization_110_input         
p 

 
ф ",б)
"і
tensor_0         
џ к
.__inference_sequential_34_layer_call_fn_583357Њ&'6789@APQRSZ[ijqrVбS
LбI
?і<
batch_normalization_110_input         
p

 
ф "!і
unknown         к
.__inference_sequential_34_layer_call_fn_583406Њ&'6789@APQRSZ[ijqrVбS
LбI
?і<
batch_normalization_110_input         
p 

 
ф "!і
unknown         у
$__inference_signature_wrapper_583534Й&'6789@APQRSZ[ijqroбl
б 
eфb
`
batch_normalization_110_input?і<
batch_normalization_110_input         "3ф0
.
dense_69"і
dense_69         