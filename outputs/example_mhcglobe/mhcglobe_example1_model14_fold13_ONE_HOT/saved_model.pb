��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18��

�
RMSprop/output/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameRMSprop/output/bias/mg
}
*RMSprop/output/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/output/bias/mg*
_output_shapes
:*
dtype0
�
RMSprop/output/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameRMSprop/output/kernel/mg
�
,RMSprop/output/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/mg*
_output_shapes
:	�*
dtype0
}
RMSprop/d3/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameRMSprop/d3/bias/mg
v
&RMSprop/d3/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d3/bias/mg*
_output_shapes	
:�*
dtype0
�
RMSprop/d3/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*%
shared_nameRMSprop/d3/kernel/mg

(RMSprop/d3/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d3/kernel/mg* 
_output_shapes
:
�	�*
dtype0
}
RMSprop/d2/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameRMSprop/d2/bias/mg
v
&RMSprop/d2/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/mg*
_output_shapes	
:�*
dtype0
�
RMSprop/d2/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameRMSprop/d2/kernel/mg

(RMSprop/d2/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/mg* 
_output_shapes
:
��*
dtype0
}
RMSprop/d1/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameRMSprop/d1/bias/mg
v
&RMSprop/d1/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/mg*
_output_shapes	
:�*
dtype0
�
RMSprop/d1/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameRMSprop/d1/kernel/mg

(RMSprop/d1/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/mg* 
_output_shapes
:
��*
dtype0
�
RMSprop/output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/output/bias/momentum
�
0RMSprop/output/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/output/bias/momentum*
_output_shapes
:*
dtype0
�
RMSprop/output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name RMSprop/output/kernel/momentum
�
2RMSprop/output/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/momentum*
_output_shapes
:	�*
dtype0
�
RMSprop/d3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameRMSprop/d3/bias/momentum
�
,RMSprop/d3/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d3/bias/momentum*
_output_shapes	
:�*
dtype0
�
RMSprop/d3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*+
shared_nameRMSprop/d3/kernel/momentum
�
.RMSprop/d3/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d3/kernel/momentum* 
_output_shapes
:
�	�*
dtype0
�
RMSprop/d2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameRMSprop/d2/bias/momentum
�
,RMSprop/d2/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/momentum*
_output_shapes	
:�*
dtype0
�
RMSprop/d2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*+
shared_nameRMSprop/d2/kernel/momentum
�
.RMSprop/d2/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
RMSprop/d1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameRMSprop/d1/bias/momentum
�
,RMSprop/d1/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/momentum*
_output_shapes	
:�*
dtype0
�
RMSprop/d1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*+
shared_nameRMSprop/d1/kernel/momentum
�
.RMSprop/d1/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
RMSprop/output/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/output/bias/rms

+RMSprop/output/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/output/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**
shared_nameRMSprop/output/kernel/rms
�
-RMSprop/output/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/rms*
_output_shapes
:	�*
dtype0

RMSprop/d3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameRMSprop/d3/bias/rms
x
'RMSprop/d3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d3/bias/rms*
_output_shapes	
:�*
dtype0
�
RMSprop/d3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*&
shared_nameRMSprop/d3/kernel/rms
�
)RMSprop/d3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d3/kernel/rms* 
_output_shapes
:
�	�*
dtype0

RMSprop/d2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameRMSprop/d2/bias/rms
x
'RMSprop/d2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/rms*
_output_shapes	
:�*
dtype0
�
RMSprop/d2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameRMSprop/d2/kernel/rms
�
)RMSprop/d2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/rms* 
_output_shapes
:
��*
dtype0

RMSprop/d1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameRMSprop/d1/bias/rms
x
'RMSprop/d1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/rms*
_output_shapes	
:�*
dtype0
�
RMSprop/d1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameRMSprop/d1/kernel/rms
�
)RMSprop/d1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/rms* 
_output_shapes
:
��*
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
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	�*
dtype0
g
d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d3/bias
`
d3/bias/Read/ReadVariableOpReadVariableOpd3/bias*
_output_shapes	
:�*
dtype0
p
	d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*
shared_name	d3/kernel
i
d3/kernel/Read/ReadVariableOpReadVariableOp	d3/kernel* 
_output_shapes
:
�	�*
dtype0
g
d2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d2/bias
`
d2/bias/Read/ReadVariableOpReadVariableOpd2/bias*
_output_shapes	
:�*
dtype0
p
	d2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name	d2/kernel
i
d2/kernel/Read/ReadVariableOpReadVariableOp	d2/kernel* 
_output_shapes
:
��*
dtype0
g
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d1/bias
`
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes	
:�*
dtype0
p
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name	d1/kernel
i
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel* 
_output_shapes
:
��*
dtype0

NoOpNoOp
�j
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�i
value�iB�i B�i
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer-7
	layer_with_weights-1
	layer-8

layer-9
layer-10
layer_with_weights-2
layer-11
layer-12
layer_with_weights-3
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
'
#_self_saveable_object_factories* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
#!_self_saveable_object_factories* 
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
#(_self_saveable_object_factories* 
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
#8_self_saveable_object_factories*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator
#@_self_saveable_object_factories* 
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
#G_self_saveable_object_factories* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
#P_self_saveable_object_factories*
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator
#X_self_saveable_object_factories* 
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
#__self_saveable_object_factories* 
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
#h_self_saveable_object_factories*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
o_random_generator
#p_self_saveable_object_factories* 
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias
#y_self_saveable_object_factories*
<
60
71
N2
O3
f4
g5
w6
x7*
<
60
71
N2
O3
f4
g5
w6
x7*

z0
{1* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
	�iter

�decay
�learning_rate
�momentum
�rho
6rms�
7rms�
Nrms�
Orms�
frms�
grms�
wrms�
xrms�6momentum�7momentum�Nmomentum�Omomentum�fmomentum�gmomentum�wmomentum�xmomentum�	6mg�	7mg�	Nmg�	Omg�	fmg�	gmg�	wmg�	xmg�*

�serving_default* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

60
71*

60
71*
	
z0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

N0
O1*

N0
O1*
	
{0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	d2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

f0
g1*

f0
g1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
YS
VARIABLE_VALUE	d3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 

w0
x1*

w0
x1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

�trace_0* 

�trace_0* 
* 
j
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
13*

�0*
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
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
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
	
z0* 
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
	
{0* 
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
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUERMSprop/d1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUERMSprop/d1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUERMSprop/d2/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUERMSprop/d2/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUERMSprop/d3/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUERMSprop/d3/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/output/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUERMSprop/output/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/d1/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/d1/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/d2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/d2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/d3/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/d3/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/output/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUERMSprop/output/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUERMSprop/d1/kernel/mgSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUERMSprop/d1/bias/mgQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUERMSprop/d2/kernel/mgSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUERMSprop/d2/bias/mgQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUERMSprop/d3/kernel/mgSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUERMSprop/d3/bias/mgQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUERMSprop/output/kernel/mgSlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUERMSprop/output/bias/mgQlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_mhc_inputPlaceholder*+
_output_shapes
:���������"*
dtype0* 
shape:���������"
�
serving_default_peptide_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_mhc_inputserving_default_peptide_input	d1/kerneld1/bias	d2/kerneld2/bias	d3/kerneld3/biasoutput/kerneloutput/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_13601
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd2/kernel/Read/ReadVariableOpd2/bias/Read/ReadVariableOpd3/kernel/Read/ReadVariableOpd3/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)RMSprop/d1/kernel/rms/Read/ReadVariableOp'RMSprop/d1/bias/rms/Read/ReadVariableOp)RMSprop/d2/kernel/rms/Read/ReadVariableOp'RMSprop/d2/bias/rms/Read/ReadVariableOp)RMSprop/d3/kernel/rms/Read/ReadVariableOp'RMSprop/d3/bias/rms/Read/ReadVariableOp-RMSprop/output/kernel/rms/Read/ReadVariableOp+RMSprop/output/bias/rms/Read/ReadVariableOp.RMSprop/d1/kernel/momentum/Read/ReadVariableOp,RMSprop/d1/bias/momentum/Read/ReadVariableOp.RMSprop/d2/kernel/momentum/Read/ReadVariableOp,RMSprop/d2/bias/momentum/Read/ReadVariableOp.RMSprop/d3/kernel/momentum/Read/ReadVariableOp,RMSprop/d3/bias/momentum/Read/ReadVariableOp2RMSprop/output/kernel/momentum/Read/ReadVariableOp0RMSprop/output/bias/momentum/Read/ReadVariableOp(RMSprop/d1/kernel/mg/Read/ReadVariableOp&RMSprop/d1/bias/mg/Read/ReadVariableOp(RMSprop/d2/kernel/mg/Read/ReadVariableOp&RMSprop/d2/bias/mg/Read/ReadVariableOp(RMSprop/d3/kernel/mg/Read/ReadVariableOp&RMSprop/d3/bias/mg/Read/ReadVariableOp,RMSprop/output/kernel/mg/Read/ReadVariableOp*RMSprop/output/bias/mg/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_14211
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d2/kerneld2/bias	d3/kerneld3/biasoutput/kerneloutput/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/d1/kernel/rmsRMSprop/d1/bias/rmsRMSprop/d2/kernel/rmsRMSprop/d2/bias/rmsRMSprop/d3/kernel/rmsRMSprop/d3/bias/rmsRMSprop/output/kernel/rmsRMSprop/output/bias/rmsRMSprop/d1/kernel/momentumRMSprop/d1/bias/momentumRMSprop/d2/kernel/momentumRMSprop/d2/bias/momentumRMSprop/d3/kernel/momentumRMSprop/d3/bias/momentumRMSprop/output/kernel/momentumRMSprop/output/bias/momentumRMSprop/d1/kernel/mgRMSprop/d1/bias/mgRMSprop/d2/kernel/mgRMSprop/d2/bias/mgRMSprop/d3/kernel/mgRMSprop/d3/bias/mgRMSprop/output/kernel/mgRMSprop/output/bias/mg*3
Tin,
*2(*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_14338��
�	
d
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13310

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
d
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13270

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_output_layer_call_and_return_conditional_losses_13158

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
d
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13230

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_13828

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
j
@__inference_skip2_layer_call_and_return_conditional_losses_13121

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13071

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�3
�
 __inference__wrapped_model_13007
	mhc_input
peptide_input;
'model_d1_matmul_readvariableop_resource:
��7
(model_d1_biasadd_readvariableop_resource:	�;
'model_d2_matmul_readvariableop_resource:
��7
(model_d2_biasadd_readvariableop_resource:	�;
'model_d3_matmul_readvariableop_resource:
�	�7
(model_d3_biasadd_readvariableop_resource:	�>
+model_output_matmul_readvariableop_resource:	�:
,model_output_biasadd_readvariableop_resource:
identity��model/d1/BiasAdd/ReadVariableOp�model/d1/MatMul/ReadVariableOp�model/d2/BiasAdd/ReadVariableOp�model/d2/MatMul/ReadVariableOp�model/d3/BiasAdd/ReadVariableOp�model/d3/MatMul/ReadVariableOp�#model/output/BiasAdd/ReadVariableOp�"model/output/MatMul/ReadVariableOpf
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
model/flatten_1/ReshapeReshape	mhc_inputmodel/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  �
model/flatten/ReshapeReshapepeptide_inputmodel/flatten/Const:output:0*
T0*(
_output_shapes
:����������Z
model/concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concat/concatConcatV2 model/flatten_1/Reshape:output:0model/flatten/Reshape:output:0!model/concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
model/d1/MatMul/ReadVariableOpReadVariableOp'model_d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/d1/MatMulMatMulmodel/concat/concat:output:0&model/d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/d1/BiasAdd/ReadVariableOpReadVariableOp(model_d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/d1/BiasAddBiasAddmodel/d1/MatMul:product:0'model/d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
model/d1/ReluRelumodel/d1/BiasAdd:output:0*
T0*(
_output_shapes
:����������u
model/dropout_d1/IdentityIdentitymodel/d1/Relu:activations:0*
T0*(
_output_shapes
:����������Y
model/skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/skip1/concatConcatV2model/concat/concat:output:0"model/dropout_d1/Identity:output:0 model/skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
model/d2/MatMul/ReadVariableOpReadVariableOp'model_d2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/d2/MatMulMatMulmodel/skip1/concat:output:0&model/d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/d2/BiasAdd/ReadVariableOpReadVariableOp(model_d2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/d2/BiasAddBiasAddmodel/d2/MatMul:product:0'model/d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
model/d2/ReluRelumodel/d2/BiasAdd:output:0*
T0*(
_output_shapes
:����������u
model/dropout_d2/IdentityIdentitymodel/d2/Relu:activations:0*
T0*(
_output_shapes
:����������Y
model/skip2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/skip2/concatConcatV2model/skip1/concat:output:0"model/dropout_d2/Identity:output:0 model/skip2/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	�
model/d3/MatMul/ReadVariableOpReadVariableOp'model_d3_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0�
model/d3/MatMulMatMulmodel/skip2/concat:output:0&model/d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/d3/BiasAdd/ReadVariableOpReadVariableOp(model_d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/d3/BiasAddBiasAddmodel/d3/MatMul:product:0'model/d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
model/d3/ReluRelumodel/d3/BiasAdd:output:0*
T0*(
_output_shapes
:����������u
model/dropout_d3/IdentityIdentitymodel/d3/Relu:activations:0*
T0*(
_output_shapes
:�����������
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/output/MatMulMatMul"model/dropout_d3/Identity:output:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
model/output/SigmoidSigmoidmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:���������g
IdentityIdentitymodel/output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^model/d1/BiasAdd/ReadVariableOp^model/d1/MatMul/ReadVariableOp ^model/d2/BiasAdd/ReadVariableOp^model/d2/MatMul/ReadVariableOp ^model/d3/BiasAdd/ReadVariableOp^model/d3/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 2B
model/d1/BiasAdd/ReadVariableOpmodel/d1/BiasAdd/ReadVariableOp2@
model/d1/MatMul/ReadVariableOpmodel/d1/MatMul/ReadVariableOp2B
model/d2/BiasAdd/ReadVariableOpmodel/d2/BiasAdd/ReadVariableOp2@
model/d2/MatMul/ReadVariableOpmodel/d2/MatMul/ReadVariableOp2B
model/d3/BiasAdd/ReadVariableOpmodel/d3/BiasAdd/ReadVariableOp2@
model/d3/MatMul/ReadVariableOpmodel/d3/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:V R
+
_output_shapes
:���������"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:���������
'
_user_specified_namepeptide_input
�

�
%__inference_model_layer_call_fn_13661
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
�	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_13416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
E
)__inference_flatten_1_layer_call_fn_13811

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_13022a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������":S O
+
_output_shapes
:���������"
 
_user_specified_nameinputs
�
�
"__inference_d2_layer_call_fn_13918

inputs
unknown:
��
	unknown_0:	�
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
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_13101p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_14070B
.kernel_regularizer_abs_readvariableop_resource:
��
identity��%kernel/Regularizer/Abs/ReadVariableOp]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/add:z:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp
�
F
*__inference_dropout_d1_layer_call_fn_13874

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
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13071a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
l
@__inference_skip2_layer_call_and_return_conditional_losses_13977
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
C
'__inference_flatten_layer_call_fn_13822

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13030a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14012

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13952

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_output_layer_call_and_return_conditional_losses_14044

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13112

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
R
&__inference_concat_layer_call_fn_13834
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_13039a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_13030

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
=__inference_d2_layer_call_and_return_conditional_losses_13101

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�%kernel/Regularizer/Abs/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp&^kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
=__inference_d1_layer_call_and_return_conditional_losses_13060

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�%kernel/Regularizer/Abs/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp&^kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
%__inference_model_layer_call_fn_13639
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
�	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_13181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
F
*__inference_dropout_d2_layer_call_fn_13942

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
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13112a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
=__inference_d3_layer_call_and_return_conditional_losses_13134

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_13817

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������":S O
+
_output_shapes
:���������"
 
_user_specified_nameinputs
�
Q
%__inference_skip1_layer_call_fn_13902
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_13080a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
c
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13884

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_d2_layer_call_fn_13947

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
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13270p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
d
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14024

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�A
�
@__inference_model_layer_call_and_return_conditional_losses_13416

inputs
inputs_1
d1_13374:
��
d1_13376:	�
d2_13381:
��
d2_13383:	�
d3_13388:
�	�
d3_13390:	�
output_13394:	�
output_13396:
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�"dropout_d1/StatefulPartitionedCall�"dropout_d2/StatefulPartitionedCall�"dropout_d3/StatefulPartitionedCall�%kernel/Regularizer/Abs/ReadVariableOp�'kernel/Regularizer_1/Abs/ReadVariableOp�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_13022�
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13030�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_13039�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_13374d1_13376*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_13060�
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13310�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_13080�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_13381d2_13383*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_13101�
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13270�
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0+dropout_d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip2_layer_call_and_return_conditional_losses_13121�
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0d3_13388d3_13390*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_13134�
"dropout_d3/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0#^dropout_d2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13230�
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d3/StatefulPartitionedCall:output:0output_13394output_13396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_13158]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    p
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpd1_13374* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
'kernel/Regularizer_1/Abs/ReadVariableOpReadVariableOpd2_13381* 
_output_shapes
:
��*
dtype0{
kernel/Regularizer_1/AbsAbs/kernel/Regularizer_1/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��m
kernel/Regularizer_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Abs:y:0%kernel/Regularizer_1/Const_1:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
kernel/Regularizer_1/addAddV2#kernel/Regularizer_1/Const:output:0kernel/Regularizer_1/mul:z:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall#^dropout_d3/StatefulPartitionedCall&^kernel/Regularizer/Abs/ReadVariableOp(^kernel/Regularizer_1/Abs/ReadVariableOp^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2H
"dropout_d3/StatefulPartitionedCall"dropout_d3/StatefulPartitionedCall2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp2R
'kernel/Regularizer_1/Abs/ReadVariableOp'kernel/Regularizer_1/Abs/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������"
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
l
@__inference_skip1_layer_call_and_return_conditional_losses_13909
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
c
*__inference_dropout_d3_layer_call_fn_14007

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13230p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_output_layer_call_fn_14033

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_13158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_d3_layer_call_fn_14002

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13145a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
%__inference_model_layer_call_fn_13457
	mhc_input
peptide_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
�	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_13416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:���������"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:���������
'
_user_specified_namepeptide_input
�
m
A__inference_concat_layer_call_and_return_conditional_losses_13841
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
__inference_loss_fn_0_14057B
.kernel_regularizer_abs_readvariableop_resource:
��
identity��%kernel/Regularizer/Abs/ReadVariableOp]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOp.kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/add:z:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp
�
�
=__inference_d1_layer_call_and_return_conditional_losses_13869

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�%kernel/Regularizer/Abs/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp&^kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
@__inference_model_layer_call_and_return_conditional_losses_13506
	mhc_input
peptide_input
d1_13464:
��
d1_13466:	�
d2_13471:
��
d2_13473:	�
d3_13478:
�	�
d3_13480:	�
output_13484:	�
output_13486:
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�%kernel/Regularizer/Abs/ReadVariableOp�'kernel/Regularizer_1/Abs/ReadVariableOp�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_13022�
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13030�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_13039�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_13464d1_13466*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_13060�
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13071�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_13080�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_13471d2_13473*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_13101�
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13112�
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0#dropout_d2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip2_layer_call_and_return_conditional_losses_13121�
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0d3_13478d3_13480*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_13134�
dropout_d3/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13145�
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d3/PartitionedCall:output:0output_13484output_13486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_13158]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    p
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpd1_13464* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
'kernel/Regularizer_1/Abs/ReadVariableOpReadVariableOpd2_13471* 
_output_shapes
:
��*
dtype0{
kernel/Regularizer_1/AbsAbs/kernel/Regularizer_1/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��m
kernel/Regularizer_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Abs:y:0%kernel/Regularizer_1/Const_1:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
kernel/Regularizer_1/addAddV2#kernel/Regularizer_1/Const:output:0kernel/Regularizer_1/mul:z:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall&^kernel/Regularizer/Abs/ReadVariableOp(^kernel/Regularizer_1/Abs/ReadVariableOp^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp2R
'kernel/Regularizer_1/Abs/ReadVariableOp'kernel/Regularizer_1/Abs/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
+
_output_shapes
:���������"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:���������
'
_user_specified_namepeptide_input
��
�
!__inference__traced_restore_14338
file_prefix.
assignvariableop_d1_kernel:
��)
assignvariableop_1_d1_bias:	�0
assignvariableop_2_d2_kernel:
��)
assignvariableop_3_d2_bias:	�0
assignvariableop_4_d3_kernel:
�	�)
assignvariableop_5_d3_bias:	�3
 assignvariableop_6_output_kernel:	�,
assignvariableop_7_output_bias:)
assignvariableop_8_rmsprop_iter:	 *
 assignvariableop_9_rmsprop_decay: 3
)assignvariableop_10_rmsprop_learning_rate: .
$assignvariableop_11_rmsprop_momentum: )
assignvariableop_12_rmsprop_rho: #
assignvariableop_13_total: #
assignvariableop_14_count: =
)assignvariableop_15_rmsprop_d1_kernel_rms:
��6
'assignvariableop_16_rmsprop_d1_bias_rms:	�=
)assignvariableop_17_rmsprop_d2_kernel_rms:
��6
'assignvariableop_18_rmsprop_d2_bias_rms:	�=
)assignvariableop_19_rmsprop_d3_kernel_rms:
�	�6
'assignvariableop_20_rmsprop_d3_bias_rms:	�@
-assignvariableop_21_rmsprop_output_kernel_rms:	�9
+assignvariableop_22_rmsprop_output_bias_rms:B
.assignvariableop_23_rmsprop_d1_kernel_momentum:
��;
,assignvariableop_24_rmsprop_d1_bias_momentum:	�B
.assignvariableop_25_rmsprop_d2_kernel_momentum:
��;
,assignvariableop_26_rmsprop_d2_bias_momentum:	�B
.assignvariableop_27_rmsprop_d3_kernel_momentum:
�	�;
,assignvariableop_28_rmsprop_d3_bias_momentum:	�E
2assignvariableop_29_rmsprop_output_kernel_momentum:	�>
0assignvariableop_30_rmsprop_output_bias_momentum:<
(assignvariableop_31_rmsprop_d1_kernel_mg:
��5
&assignvariableop_32_rmsprop_d1_bias_mg:	�<
(assignvariableop_33_rmsprop_d2_kernel_mg:
��5
&assignvariableop_34_rmsprop_d2_bias_mg:	�<
(assignvariableop_35_rmsprop_d3_kernel_mg:
�	�5
&assignvariableop_36_rmsprop_d3_bias_mg:	�?
,assignvariableop_37_rmsprop_output_kernel_mg:	�8
*assignvariableop_38_rmsprop_output_bias_mg:
identity_40��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_d2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_d2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_d3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_d3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_rmsprop_rhoIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_rmsprop_d1_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_rmsprop_d1_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_rmsprop_d2_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_rmsprop_d2_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_rmsprop_d3_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_rmsprop_d3_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_rmsprop_output_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_rmsprop_output_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp.assignvariableop_23_rmsprop_d1_kernel_momentumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_rmsprop_d1_bias_momentumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_rmsprop_d2_kernel_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_rmsprop_d2_bias_momentumIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_rmsprop_d3_kernel_momentumIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_rmsprop_d3_bias_momentumIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp2assignvariableop_29_rmsprop_output_kernel_momentumIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp0assignvariableop_30_rmsprop_output_bias_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_rmsprop_d1_kernel_mgIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_rmsprop_d1_bias_mgIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_rmsprop_d2_kernel_mgIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_rmsprop_d2_bias_mgIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_rmsprop_d3_kernel_mgIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_rmsprop_d3_bias_mgIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp,assignvariableop_37_rmsprop_output_kernel_mgIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_rmsprop_output_bias_mgIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�?
�
@__inference_model_layer_call_and_return_conditional_losses_13723
inputs_0
inputs_15
!d1_matmul_readvariableop_resource:
��1
"d1_biasadd_readvariableop_resource:	�5
!d2_matmul_readvariableop_resource:
��1
"d2_biasadd_readvariableop_resource:	�5
!d3_matmul_readvariableop_resource:
�	�1
"d3_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity��d1/BiasAdd/ReadVariableOp�d1/MatMul/ReadVariableOp�d2/BiasAdd/ReadVariableOp�d2/MatMul/ReadVariableOp�d3/BiasAdd/ReadVariableOp�d3/MatMul/ReadVariableOp�%kernel/Regularizer/Abs/ReadVariableOp�'kernel/Regularizer_1/Abs/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  s
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  o
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:����������T
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:����������i
dropout_d1/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:����������S
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
skip1/concatConcatV2concat/concat:output:0dropout_d1/Identity:output:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������|
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:����������i
dropout_d2/IdentityIdentityd2/Relu:activations:0*
T0*(
_output_shapes
:����������S
skip2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
skip2/concatConcatV2skip1/concat:output:0dropout_d2/Identity:output:0skip2/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	|
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0
	d3/MatMulMatMulskip2/concat:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:����������i
dropout_d3/IdentityIdentityd3/Relu:activations:0*
T0*(
_output_shapes
:�����������
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output/MatMulMatMuldropout_d3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'kernel/Regularizer_1/Abs/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0{
kernel/Regularizer_1/AbsAbs/kernel/Regularizer_1/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��m
kernel/Regularizer_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Abs:y:0%kernel/Regularizer_1/Const_1:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
kernel/Regularizer_1/addAddV2#kernel/Regularizer_1/Const:output:0kernel/Regularizer_1/mul:z:0*
T0*
_output_shapes
: a
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp&^kernel/Regularizer/Abs/ReadVariableOp(^kernel/Regularizer_1/Abs/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp2R
'kernel/Regularizer_1/Abs/ReadVariableOp'kernel/Regularizer_1/Abs/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:U Q
+
_output_shapes
:���������"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�V
�
@__inference_model_layer_call_and_return_conditional_losses_13806
inputs_0
inputs_15
!d1_matmul_readvariableop_resource:
��1
"d1_biasadd_readvariableop_resource:	�5
!d2_matmul_readvariableop_resource:
��1
"d2_biasadd_readvariableop_resource:	�5
!d3_matmul_readvariableop_resource:
�	�1
"d3_biasadd_readvariableop_resource:	�8
%output_matmul_readvariableop_resource:	�4
&output_biasadd_readvariableop_resource:
identity��d1/BiasAdd/ReadVariableOp�d1/MatMul/ReadVariableOp�d2/BiasAdd/ReadVariableOp�d2/MatMul/ReadVariableOp�d3/BiasAdd/ReadVariableOp�d3/MatMul/ReadVariableOp�%kernel/Regularizer/Abs/ReadVariableOp�'kernel/Regularizer_1/Abs/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  s
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  o
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:����������T
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:����������]
dropout_d1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_d1/dropout/MulMuld1/Relu:activations:0!dropout_d1/dropout/Const:output:0*
T0*(
_output_shapes
:����������]
dropout_d1/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_d1/dropout/random_uniform/RandomUniformRandomUniform!dropout_d1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_d1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_d1/dropout/GreaterEqualGreaterEqual8dropout_d1/dropout/random_uniform/RandomUniform:output:0*dropout_d1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_d1/dropout/CastCast#dropout_d1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_d1/dropout/Mul_1Muldropout_d1/dropout/Mul:z:0dropout_d1/dropout/Cast:y:0*
T0*(
_output_shapes
:����������S
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
skip1/concatConcatV2concat/concat:output:0dropout_d1/dropout/Mul_1:z:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������|
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:����������]
dropout_d2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_d2/dropout/MulMuld2/Relu:activations:0!dropout_d2/dropout/Const:output:0*
T0*(
_output_shapes
:����������]
dropout_d2/dropout/ShapeShaped2/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_d2/dropout/random_uniform/RandomUniformRandomUniform!dropout_d2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_d2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_d2/dropout/GreaterEqualGreaterEqual8dropout_d2/dropout/random_uniform/RandomUniform:output:0*dropout_d2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_d2/dropout/CastCast#dropout_d2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_d2/dropout/Mul_1Muldropout_d2/dropout/Mul:z:0dropout_d2/dropout/Cast:y:0*
T0*(
_output_shapes
:����������S
skip2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
skip2/concatConcatV2skip1/concat:output:0dropout_d2/dropout/Mul_1:z:0skip2/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	|
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0
	d3/MatMulMatMulskip2/concat:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:����������]
dropout_d3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
dropout_d3/dropout/MulMuld3/Relu:activations:0!dropout_d3/dropout/Const:output:0*
T0*(
_output_shapes
:����������]
dropout_d3/dropout/ShapeShaped3/Relu:activations:0*
T0*
_output_shapes
:�
/dropout_d3/dropout/random_uniform/RandomUniformRandomUniform!dropout_d3/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0f
!dropout_d3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout_d3/dropout/GreaterEqualGreaterEqual8dropout_d3/dropout/random_uniform/RandomUniform:output:0*dropout_d3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_d3/dropout/CastCast#dropout_d3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_d3/dropout/Mul_1Muldropout_d3/dropout/Mul:z:0dropout_d3/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
output/MatMulMatMuldropout_d3/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
'kernel/Regularizer_1/Abs/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0{
kernel/Regularizer_1/AbsAbs/kernel/Regularizer_1/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��m
kernel/Regularizer_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Abs:y:0%kernel/Regularizer_1/Const_1:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
kernel/Regularizer_1/addAddV2#kernel/Regularizer_1/Const:output:0kernel/Regularizer_1/mul:z:0*
T0*
_output_shapes
: a
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp&^kernel/Regularizer/Abs/ReadVariableOp(^kernel/Regularizer_1/Abs/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp2R
'kernel/Regularizer_1/Abs/ReadVariableOp'kernel/Regularizer_1/Abs/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:U Q
+
_output_shapes
:���������"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
=__inference_d3_layer_call_and_return_conditional_losses_13997

inputs2
matmul_readvariableop_resource:
�	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�

�
%__inference_model_layer_call_fn_13200
	mhc_input
peptide_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
�	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_13181o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:���������"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:���������
'
_user_specified_namepeptide_input
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_13022

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������":S O
+
_output_shapes
:���������"
 
_user_specified_nameinputs
�
c
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13145

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
Q
%__inference_skip2_layer_call_fn_13970
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip2_layer_call_and_return_conditional_losses_13121a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
"__inference_d1_layer_call_fn_13850

inputs
unknown:
��
	unknown_0:	�
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
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_13060p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
A__inference_concat_layer_call_and_return_conditional_losses_13039

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
=__inference_d2_layer_call_and_return_conditional_losses_13937

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�%kernel/Regularizer/Abs/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp&^kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
d
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13896

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�S
�
__inference__traced_save_14211
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop(
$savev2_d2_kernel_read_readvariableop&
"savev2_d2_bias_read_readvariableop(
$savev2_d3_kernel_read_readvariableop&
"savev2_d3_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_rmsprop_d1_kernel_rms_read_readvariableop2
.savev2_rmsprop_d1_bias_rms_read_readvariableop4
0savev2_rmsprop_d2_kernel_rms_read_readvariableop2
.savev2_rmsprop_d2_bias_rms_read_readvariableop4
0savev2_rmsprop_d3_kernel_rms_read_readvariableop2
.savev2_rmsprop_d3_bias_rms_read_readvariableop8
4savev2_rmsprop_output_kernel_rms_read_readvariableop6
2savev2_rmsprop_output_bias_rms_read_readvariableop9
5savev2_rmsprop_d1_kernel_momentum_read_readvariableop7
3savev2_rmsprop_d1_bias_momentum_read_readvariableop9
5savev2_rmsprop_d2_kernel_momentum_read_readvariableop7
3savev2_rmsprop_d2_bias_momentum_read_readvariableop9
5savev2_rmsprop_d3_kernel_momentum_read_readvariableop7
3savev2_rmsprop_d3_bias_momentum_read_readvariableop=
9savev2_rmsprop_output_kernel_momentum_read_readvariableop;
7savev2_rmsprop_output_bias_momentum_read_readvariableop3
/savev2_rmsprop_d1_kernel_mg_read_readvariableop1
-savev2_rmsprop_d1_bias_mg_read_readvariableop3
/savev2_rmsprop_d2_kernel_mg_read_readvariableop1
-savev2_rmsprop_d2_bias_mg_read_readvariableop3
/savev2_rmsprop_d3_kernel_mg_read_readvariableop1
-savev2_rmsprop_d3_bias_mg_read_readvariableop7
3savev2_rmsprop_output_kernel_mg_read_readvariableop5
1savev2_rmsprop_output_bias_mg_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d2_kernel_read_readvariableop"savev2_d2_bias_read_readvariableop$savev2_d3_kernel_read_readvariableop"savev2_d3_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_rmsprop_d1_kernel_rms_read_readvariableop.savev2_rmsprop_d1_bias_rms_read_readvariableop0savev2_rmsprop_d2_kernel_rms_read_readvariableop.savev2_rmsprop_d2_bias_rms_read_readvariableop0savev2_rmsprop_d3_kernel_rms_read_readvariableop.savev2_rmsprop_d3_bias_rms_read_readvariableop4savev2_rmsprop_output_kernel_rms_read_readvariableop2savev2_rmsprop_output_bias_rms_read_readvariableop5savev2_rmsprop_d1_kernel_momentum_read_readvariableop3savev2_rmsprop_d1_bias_momentum_read_readvariableop5savev2_rmsprop_d2_kernel_momentum_read_readvariableop3savev2_rmsprop_d2_bias_momentum_read_readvariableop5savev2_rmsprop_d3_kernel_momentum_read_readvariableop3savev2_rmsprop_d3_bias_momentum_read_readvariableop9savev2_rmsprop_output_kernel_momentum_read_readvariableop7savev2_rmsprop_output_bias_momentum_read_readvariableop/savev2_rmsprop_d1_kernel_mg_read_readvariableop-savev2_rmsprop_d1_bias_mg_read_readvariableop/savev2_rmsprop_d2_kernel_mg_read_readvariableop-savev2_rmsprop_d2_bias_mg_read_readvariableop/savev2_rmsprop_d3_kernel_mg_read_readvariableop-savev2_rmsprop_d3_bias_mg_read_readvariableop3savev2_rmsprop_output_kernel_mg_read_readvariableop1savev2_rmsprop_output_bias_mg_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
��:�:
�	�:�:	�:: : : : : : : :
��:�:
��:�:
�	�:�:	�::
��:�:
��:�:
�	�:�:	�::
��:�:
��:�:
�	�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
�	�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::& "
 
_output_shapes
:
��:!!

_output_shapes	
:�:&""
 
_output_shapes
:
��:!#

_output_shapes	
:�:&$"
 
_output_shapes
:
�	�:!%

_output_shapes	
:�:%&!

_output_shapes
:	�: '

_output_shapes
::(

_output_shapes
: 
�	
d
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13964

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_d3_layer_call_fn_13986

inputs
unknown:
�	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_13134p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�

�
#__inference_signature_wrapper_13601
	mhc_input
peptide_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
�	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_13007o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:���������"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:���������
'
_user_specified_namepeptide_input
�
j
@__inference_skip1_layer_call_and_return_conditional_losses_13080

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_d1_layer_call_fn_13879

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
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13310p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�B
�
@__inference_model_layer_call_and_return_conditional_losses_13555
	mhc_input
peptide_input
d1_13513:
��
d1_13515:	�
d2_13520:
��
d2_13522:	�
d3_13527:
�	�
d3_13529:	�
output_13533:	�
output_13535:
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�"dropout_d1/StatefulPartitionedCall�"dropout_d2/StatefulPartitionedCall�"dropout_d3/StatefulPartitionedCall�%kernel/Regularizer/Abs/ReadVariableOp�'kernel/Regularizer_1/Abs/ReadVariableOp�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_13022�
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13030�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_13039�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_13513d1_13515*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_13060�
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13310�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_13080�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_13520d2_13522*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_13101�
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13270�
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0+dropout_d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip2_layer_call_and_return_conditional_losses_13121�
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0d3_13527d3_13529*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_13134�
"dropout_d3/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0#^dropout_d2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13230�
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d3/StatefulPartitionedCall:output:0output_13533output_13535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_13158]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    p
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpd1_13513* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
'kernel/Regularizer_1/Abs/ReadVariableOpReadVariableOpd2_13520* 
_output_shapes
:
��*
dtype0{
kernel/Regularizer_1/AbsAbs/kernel/Regularizer_1/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��m
kernel/Regularizer_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Abs:y:0%kernel/Regularizer_1/Const_1:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
kernel/Regularizer_1/addAddV2#kernel/Regularizer_1/Const:output:0kernel/Regularizer_1/mul:z:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall#^dropout_d3/StatefulPartitionedCall&^kernel/Regularizer/Abs/ReadVariableOp(^kernel/Regularizer_1/Abs/ReadVariableOp^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2H
"dropout_d3/StatefulPartitionedCall"dropout_d3/StatefulPartitionedCall2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp2R
'kernel/Regularizer_1/Abs/ReadVariableOp'kernel/Regularizer_1/Abs/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
+
_output_shapes
:���������"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:���������
'
_user_specified_namepeptide_input
�=
�
@__inference_model_layer_call_and_return_conditional_losses_13181

inputs
inputs_1
d1_13061:
��
d1_13063:	�
d2_13102:
��
d2_13104:	�
d3_13135:
�	�
d3_13137:	�
output_13159:	�
output_13161:
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�%kernel/Regularizer/Abs/ReadVariableOp�'kernel/Regularizer_1/Abs/ReadVariableOp�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_13022�
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13030�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_13039�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_13061d1_13063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_13060�
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13071�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_13080�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_13102d2_13104*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_13101�
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13112�
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0#dropout_d2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_skip2_layer_call_and_return_conditional_losses_13121�
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0d3_13135d3_13137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_13134�
dropout_d3/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_13145�
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d3/PartitionedCall:output:0output_13159output_13161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_13158]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    p
%kernel/Regularizer/Abs/ReadVariableOpReadVariableOpd1_13061* 
_output_shapes
:
��*
dtype0w
kernel/Regularizer/AbsAbs-kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��k
kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       
kernel/Regularizer/SumSumkernel/Regularizer/Abs:y:0#kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
kernel/Regularizer/addAddV2!kernel/Regularizer/Const:output:0kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
'kernel/Regularizer_1/Abs/ReadVariableOpReadVariableOpd2_13102* 
_output_shapes
:
��*
dtype0{
kernel/Regularizer_1/AbsAbs/kernel/Regularizer_1/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��m
kernel/Regularizer_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Abs:y:0%kernel/Regularizer_1/Const_1:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
kernel/Regularizer_1/addAddV2#kernel/Regularizer_1/Const:output:0kernel/Regularizer_1/mul:z:0*
T0*
_output_shapes
: v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall&^kernel/Regularizer/Abs/ReadVariableOp(^kernel/Regularizer_1/Abs/ReadVariableOp^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������":���������: : : : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2N
%kernel/Regularizer/Abs/ReadVariableOp%kernel/Regularizer/Abs/ReadVariableOp2R
'kernel/Regularizer_1/Abs/ReadVariableOp'kernel/Regularizer_1/Abs/ReadVariableOp2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������"
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
	mhc_input6
serving_default_mhc_input:0���������"
K
peptide_input:
serving_default_peptide_input:0���������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer-7
	layer_with_weights-1
	layer-8

layer-9
layer-10
layer_with_weights-2
layer-11
layer-12
layer_with_weights-3
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
D
#_self_saveable_object_factories"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
#!_self_saveable_object_factories"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
#(_self_saveable_object_factories"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias
#8_self_saveable_object_factories"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_random_generator
#@_self_saveable_object_factories"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses
#G_self_saveable_object_factories"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias
#P_self_saveable_object_factories"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator
#X_self_saveable_object_factories"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
#__self_saveable_object_factories"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
#h_self_saveable_object_factories"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
o_random_generator
#p_self_saveable_object_factories"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

wkernel
xbias
#y_self_saveable_object_factories"
_tf_keras_layer
X
60
71
N2
O3
f4
g5
w6
x7"
trackable_list_wrapper
X
60
71
N2
O3
f4
g5
w6
x7"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_13200
%__inference_model_layer_call_fn_13639
%__inference_model_layer_call_fn_13661
%__inference_model_layer_call_fn_13457�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
@__inference_model_layer_call_and_return_conditional_losses_13723
@__inference_model_layer_call_and_return_conditional_losses_13806
@__inference_model_layer_call_and_return_conditional_losses_13506
@__inference_model_layer_call_and_return_conditional_losses_13555�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_13007	mhc_inputpeptide_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	�iter

�decay
�learning_rate
�momentum
�rho
6rms�
7rms�
Nrms�
Orms�
frms�
grms�
wrms�
xrms�6momentum�7momentum�Nmomentum�Omomentum�fmomentum�gmomentum�wmomentum�xmomentum�	6mg�	7mg�	Nmg�	Omg�	fmg�	gmg�	wmg�	xmg�"
	optimizer
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_flatten_1_layer_call_fn_13811�
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
 z�trace_0
�
�trace_02�
D__inference_flatten_1_layer_call_and_return_conditional_losses_13817�
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
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_13822�
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
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_13828�
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
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_concat_layer_call_fn_13834�
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
 z�trace_0
�
�trace_02�
A__inference_concat_layer_call_and_return_conditional_losses_13841�
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
 z�trace_0
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
'
z0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_d1_layer_call_fn_13850�
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
 z�trace_0
�
�trace_02�
=__inference_d1_layer_call_and_return_conditional_losses_13869�
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
 z�trace_0
:
��2	d1/kernel
:�2d1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_d1_layer_call_fn_13874
*__inference_dropout_d1_layer_call_fn_13879�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13884
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13896�
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
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_skip1_layer_call_fn_13902�
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
 z�trace_0
�
�trace_02�
@__inference_skip1_layer_call_and_return_conditional_losses_13909�
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
 z�trace_0
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_d2_layer_call_fn_13918�
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
 z�trace_0
�
�trace_02�
=__inference_d2_layer_call_and_return_conditional_losses_13937�
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
 z�trace_0
:
��2	d2/kernel
:�2d2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_d2_layer_call_fn_13942
*__inference_dropout_d2_layer_call_fn_13947�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13952
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13964�
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
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_skip2_layer_call_fn_13970�
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
 z�trace_0
�
�trace_02�
@__inference_skip2_layer_call_and_return_conditional_losses_13977�
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
 z�trace_0
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
"__inference_d3_layer_call_fn_13986�
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
 z�trace_0
�
�trace_02�
=__inference_d3_layer_call_and_return_conditional_losses_13997�
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
 z�trace_0
:
�	�2	d3/kernel
:�2d3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_d3_layer_call_fn_14002
*__inference_dropout_d3_layer_call_fn_14007�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14012
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14024�
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
 z�trace_0z�trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_output_layer_call_fn_14033�
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
 z�trace_0
�
�trace_02�
A__inference_output_layer_call_and_return_conditional_losses_14044�
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
 z�trace_0
 :	�2output/kernel
:2output/bias
 "
trackable_dict_wrapper
�
�trace_02�
__inference_loss_fn_0_14057�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_14070�
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
annotations� *� z�trace_0
 "
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
13"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_13200	mhc_inputpeptide_input"�
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
�B�
%__inference_model_layer_call_fn_13639inputs/0inputs/1"�
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
�B�
%__inference_model_layer_call_fn_13661inputs/0inputs/1"�
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
�B�
%__inference_model_layer_call_fn_13457	mhc_inputpeptide_input"�
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
�B�
@__inference_model_layer_call_and_return_conditional_losses_13723inputs/0inputs/1"�
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
�B�
@__inference_model_layer_call_and_return_conditional_losses_13806inputs/0inputs/1"�
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
�B�
@__inference_model_layer_call_and_return_conditional_losses_13506	mhc_inputpeptide_input"�
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
�B�
@__inference_model_layer_call_and_return_conditional_losses_13555	mhc_inputpeptide_input"�
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
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
�B�
#__inference_signature_wrapper_13601	mhc_inputpeptide_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_flatten_1_layer_call_fn_13811inputs"�
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
�B�
D__inference_flatten_1_layer_call_and_return_conditional_losses_13817inputs"�
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
�B�
'__inference_flatten_layer_call_fn_13822inputs"�
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
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_13828inputs"�
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
�B�
&__inference_concat_layer_call_fn_13834inputs/0inputs/1"�
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
�B�
A__inference_concat_layer_call_and_return_conditional_losses_13841inputs/0inputs/1"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
z0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_d1_layer_call_fn_13850inputs"�
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
�B�
=__inference_d1_layer_call_and_return_conditional_losses_13869inputs"�
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
�B�
*__inference_dropout_d1_layer_call_fn_13874inputs"�
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
�B�
*__inference_dropout_d1_layer_call_fn_13879inputs"�
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
�B�
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13884inputs"�
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
�B�
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13896inputs"�
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
�B�
%__inference_skip1_layer_call_fn_13902inputs/0inputs/1"�
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
�B�
@__inference_skip1_layer_call_and_return_conditional_losses_13909inputs/0inputs/1"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
{0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
"__inference_d2_layer_call_fn_13918inputs"�
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
�B�
=__inference_d2_layer_call_and_return_conditional_losses_13937inputs"�
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
�B�
*__inference_dropout_d2_layer_call_fn_13942inputs"�
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
�B�
*__inference_dropout_d2_layer_call_fn_13947inputs"�
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
�B�
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13952inputs"�
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
�B�
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13964inputs"�
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
�B�
%__inference_skip2_layer_call_fn_13970inputs/0inputs/1"�
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
�B�
@__inference_skip2_layer_call_and_return_conditional_losses_13977inputs/0inputs/1"�
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
�B�
"__inference_d3_layer_call_fn_13986inputs"�
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
�B�
=__inference_d3_layer_call_and_return_conditional_losses_13997inputs"�
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
�B�
*__inference_dropout_d3_layer_call_fn_14002inputs"�
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
�B�
*__inference_dropout_d3_layer_call_fn_14007inputs"�
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
�B�
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14012inputs"�
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
�B�
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14024inputs"�
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
�B�
&__inference_output_layer_call_fn_14033inputs"�
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
�B�
A__inference_output_layer_call_and_return_conditional_losses_14044inputs"�
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
�B�
__inference_loss_fn_0_14057"�
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
�B�
__inference_loss_fn_1_14070"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
':%
��2RMSprop/d1/kernel/rms
 :�2RMSprop/d1/bias/rms
':%
��2RMSprop/d2/kernel/rms
 :�2RMSprop/d2/bias/rms
':%
�	�2RMSprop/d3/kernel/rms
 :�2RMSprop/d3/bias/rms
*:(	�2RMSprop/output/kernel/rms
#:!2RMSprop/output/bias/rms
,:*
��2RMSprop/d1/kernel/momentum
%:#�2RMSprop/d1/bias/momentum
,:*
��2RMSprop/d2/kernel/momentum
%:#�2RMSprop/d2/bias/momentum
,:*
�	�2RMSprop/d3/kernel/momentum
%:#�2RMSprop/d3/bias/momentum
/:-	�2RMSprop/output/kernel/momentum
(:&2RMSprop/output/bias/momentum
&:$
��2RMSprop/d1/kernel/mg
:�2RMSprop/d1/bias/mg
&:$
��2RMSprop/d2/kernel/mg
:�2RMSprop/d2/bias/mg
&:$
�	�2RMSprop/d3/kernel/mg
:�2RMSprop/d3/bias/mg
):'	�2RMSprop/output/kernel/mg
": 2RMSprop/output/bias/mg�
 __inference__wrapped_model_13007�67NOfgwxh�e
^�[
Y�V
'�$
	mhc_input���������"
+�(
peptide_input���������
� "/�,
*
output �
output����������
A__inference_concat_layer_call_and_return_conditional_losses_13841�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
&__inference_concat_layer_call_fn_13834y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
=__inference_d1_layer_call_and_return_conditional_losses_13869^670�-
&�#
!�
inputs����������
� "&�#
�
0����������
� w
"__inference_d1_layer_call_fn_13850Q670�-
&�#
!�
inputs����������
� "������������
=__inference_d2_layer_call_and_return_conditional_losses_13937^NO0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� w
"__inference_d2_layer_call_fn_13918QNO0�-
&�#
!�
inputs����������
� "������������
=__inference_d3_layer_call_and_return_conditional_losses_13997^fg0�-
&�#
!�
inputs����������	
� "&�#
�
0����������
� w
"__inference_d3_layer_call_fn_13986Qfg0�-
&�#
!�
inputs����������	
� "������������
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13884^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
E__inference_dropout_d1_layer_call_and_return_conditional_losses_13896^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� 
*__inference_dropout_d1_layer_call_fn_13874Q4�1
*�'
!�
inputs����������
p 
� "�����������
*__inference_dropout_d1_layer_call_fn_13879Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13952^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
E__inference_dropout_d2_layer_call_and_return_conditional_losses_13964^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� 
*__inference_dropout_d2_layer_call_fn_13942Q4�1
*�'
!�
inputs����������
p 
� "�����������
*__inference_dropout_d2_layer_call_fn_13947Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14012^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
E__inference_dropout_d3_layer_call_and_return_conditional_losses_14024^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� 
*__inference_dropout_d3_layer_call_fn_14002Q4�1
*�'
!�
inputs����������
p 
� "�����������
*__inference_dropout_d3_layer_call_fn_14007Q4�1
*�'
!�
inputs����������
p
� "������������
D__inference_flatten_1_layer_call_and_return_conditional_losses_13817]3�0
)�&
$�!
inputs���������"
� "&�#
�
0����������
� }
)__inference_flatten_1_layer_call_fn_13811P3�0
)�&
$�!
inputs���������"
� "������������
B__inference_flatten_layer_call_and_return_conditional_losses_13828]3�0
)�&
$�!
inputs���������
� "&�#
�
0����������
� {
'__inference_flatten_layer_call_fn_13822P3�0
)�&
$�!
inputs���������
� "�����������:
__inference_loss_fn_0_140576�

� 
� "� :
__inference_loss_fn_1_14070N�

� 
� "� �
@__inference_model_layer_call_and_return_conditional_losses_13506�67NOfgwxp�m
f�c
Y�V
'�$
	mhc_input���������"
+�(
peptide_input���������
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_13555�67NOfgwxp�m
f�c
Y�V
'�$
	mhc_input���������"
+�(
peptide_input���������
p

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_13723�67NOfgwxj�g
`�]
S�P
&�#
inputs/0���������"
&�#
inputs/1���������
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_13806�67NOfgwxj�g
`�]
S�P
&�#
inputs/0���������"
&�#
inputs/1���������
p

 
� "%�"
�
0���������
� �
%__inference_model_layer_call_fn_13200�67NOfgwxp�m
f�c
Y�V
'�$
	mhc_input���������"
+�(
peptide_input���������
p 

 
� "�����������
%__inference_model_layer_call_fn_13457�67NOfgwxp�m
f�c
Y�V
'�$
	mhc_input���������"
+�(
peptide_input���������
p

 
� "�����������
%__inference_model_layer_call_fn_13639�67NOfgwxj�g
`�]
S�P
&�#
inputs/0���������"
&�#
inputs/1���������
p 

 
� "�����������
%__inference_model_layer_call_fn_13661�67NOfgwxj�g
`�]
S�P
&�#
inputs/0���������"
&�#
inputs/1���������
p

 
� "�����������
A__inference_output_layer_call_and_return_conditional_losses_14044]wx0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� z
&__inference_output_layer_call_fn_14033Pwx0�-
&�#
!�
inputs����������
� "�����������
#__inference_signature_wrapper_13601�67NOfgwx��~
� 
w�t
4
	mhc_input'�$
	mhc_input���������"
<
peptide_input+�(
peptide_input���������"/�,
*
output �
output����������
@__inference_skip1_layer_call_and_return_conditional_losses_13909�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
%__inference_skip1_layer_call_fn_13902y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
@__inference_skip2_layer_call_and_return_conditional_losses_13977�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������	
� �
%__inference_skip2_layer_call_fn_13970y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "�����������	