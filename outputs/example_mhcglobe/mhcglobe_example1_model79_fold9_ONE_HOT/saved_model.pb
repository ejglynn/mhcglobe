Ï§	
Ê
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18Û 

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

RMSprop/output/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameRMSprop/output/kernel/mg

,RMSprop/output/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/mg*
_output_shapes
:	*
dtype0
}
RMSprop/d2/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameRMSprop/d2/bias/mg
v
&RMSprop/d2/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/mg*
_output_shapes	
:*
dtype0

RMSprop/d2/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô	*%
shared_nameRMSprop/d2/kernel/mg

(RMSprop/d2/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/mg* 
_output_shapes
:
Ô	*
dtype0
}
RMSprop/d1/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameRMSprop/d1/bias/mg
v
&RMSprop/d1/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/mg*
_output_shapes	
:*
dtype0

RMSprop/d1/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô*%
shared_nameRMSprop/d1/kernel/mg

(RMSprop/d1/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/mg* 
_output_shapes
:
Ô*
dtype0

RMSprop/output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/output/bias/momentum

0RMSprop/output/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/output/bias/momentum*
_output_shapes
:*
dtype0

RMSprop/output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name RMSprop/output/kernel/momentum

2RMSprop/output/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/momentum*
_output_shapes
:	*
dtype0

RMSprop/d2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/d2/bias/momentum

,RMSprop/d2/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/momentum*
_output_shapes	
:*
dtype0

RMSprop/d2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô	*+
shared_nameRMSprop/d2/kernel/momentum

.RMSprop/d2/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/momentum* 
_output_shapes
:
Ô	*
dtype0

RMSprop/d1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/d1/bias/momentum

,RMSprop/d1/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/momentum*
_output_shapes	
:*
dtype0

RMSprop/d1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô*+
shared_nameRMSprop/d1/kernel/momentum

.RMSprop/d1/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/momentum* 
_output_shapes
:
Ô*
dtype0

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

RMSprop/output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameRMSprop/output/kernel/rms

-RMSprop/output/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/rms*
_output_shapes
:	*
dtype0

RMSprop/d2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameRMSprop/d2/bias/rms
x
'RMSprop/d2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/rms*
_output_shapes	
:*
dtype0

RMSprop/d2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô	*&
shared_nameRMSprop/d2/kernel/rms

)RMSprop/d2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/rms* 
_output_shapes
:
Ô	*
dtype0

RMSprop/d1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameRMSprop/d1/bias/rms
x
'RMSprop/d1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/rms*
_output_shapes	
:*
dtype0

RMSprop/d1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô*&
shared_nameRMSprop/d1/kernel/rms

)RMSprop/d1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/rms* 
_output_shapes
:
Ô*
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
shape:	*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	*
dtype0
g
d2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d2/bias
`
d2/bias/Read/ReadVariableOpReadVariableOpd2/bias*
_output_shapes	
:*
dtype0
p
	d2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô	*
shared_name	d2/kernel
i
d2/kernel/Read/ReadVariableOpReadVariableOp	d2/kernel* 
_output_shapes
:
Ô	*
dtype0
g
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d1/bias
`
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes	
:*
dtype0
p
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô*
shared_name	d1/kernel
i
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel* 
_output_shapes
:
Ô*
dtype0

NoOpNoOp
öR
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*±R
value§RB¤R BR
Ï
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
layer_with_weights-2
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
signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
'
#_self_saveable_object_factories* 
³
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories* 
³
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
#%_self_saveable_object_factories* 
³
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
#,_self_saveable_object_factories* 
Ë
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
#5_self_saveable_object_factories*
Ê
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator
#=_self_saveable_object_factories* 
³
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
#D_self_saveable_object_factories* 
Ë
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
#M_self_saveable_object_factories*
Ê
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator
#U_self_saveable_object_factories* 
Ë
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
#^_self_saveable_object_factories*
.
30
41
K2
L3
\4
]5*
.
30
41
K2
L3
\4
]5*

_0
`1* 
°
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ftrace_0
gtrace_1
htrace_2
itrace_3* 
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
* 
³
niter
	odecay
plearning_rate
qmomentum
rrho
3rms¾
4rms¿
KrmsÀ
LrmsÁ
\rmsÂ
]rmsÃ3momentumÄ4momentumÅKmomentumÆLmomentumÇ\momentumÈ]momentumÉ	3mgÊ	4mgË	KmgÌ	LmgÍ	\mgÎ	]mgÏ*

sserving_default* 
* 
* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ytrace_0* 

ztrace_0* 
* 
* 
* 
* 

{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

30
41*

30
41*
	
_0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

trace_0* 

trace_0* 
YS
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

K0
L1*

K0
L1*
	
`0* 

 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

¥trace_0* 

¦trace_0* 
YS
VARIABLE_VALUE	d2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEd2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

¬trace_0
­trace_1* 

®trace_0
¯trace_1* 
* 
* 

\0
]1*

\0
]1*
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

µtrace_0* 

¶trace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

·trace_0* 

¸trace_0* 
* 
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

¹0*
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
	
_0* 
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
	
`0* 
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
º	variables
»	keras_api

¼total

½count*

¼0
½1*

º	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUERMSprop/d1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUERMSprop/d1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUERMSprop/d2/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUERMSprop/d2/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/output/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUERMSprop/output/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/d1/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/d1/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/d2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/d2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/output/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/output/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUERMSprop/d1/kernel/mgSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUERMSprop/d1/bias/mgQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUERMSprop/d2/kernel/mgSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUERMSprop/d2/bias/mgQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/output/kernel/mgSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUERMSprop/output/bias/mgQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_mhc_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ"

serving_default_peptide_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
£
StatefulPartitionedCallStatefulPartitionedCallserving_default_mhc_inputserving_default_peptide_input	d1/kerneld1/bias	d2/kerneld2/biasoutput/kerneloutput/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_21025
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd2/kernel/Read/ReadVariableOpd2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)RMSprop/d1/kernel/rms/Read/ReadVariableOp'RMSprop/d1/bias/rms/Read/ReadVariableOp)RMSprop/d2/kernel/rms/Read/ReadVariableOp'RMSprop/d2/bias/rms/Read/ReadVariableOp-RMSprop/output/kernel/rms/Read/ReadVariableOp+RMSprop/output/bias/rms/Read/ReadVariableOp.RMSprop/d1/kernel/momentum/Read/ReadVariableOp,RMSprop/d1/bias/momentum/Read/ReadVariableOp.RMSprop/d2/kernel/momentum/Read/ReadVariableOp,RMSprop/d2/bias/momentum/Read/ReadVariableOp2RMSprop/output/kernel/momentum/Read/ReadVariableOp0RMSprop/output/bias/momentum/Read/ReadVariableOp(RMSprop/d1/kernel/mg/Read/ReadVariableOp&RMSprop/d1/bias/mg/Read/ReadVariableOp(RMSprop/d2/kernel/mg/Read/ReadVariableOp&RMSprop/d2/bias/mg/Read/ReadVariableOp,RMSprop/output/kernel/mg/Read/ReadVariableOp*RMSprop/output/bias/mg/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_21444

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d2/kerneld2/biasoutput/kerneloutput/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/d1/kernel/rmsRMSprop/d1/bias/rmsRMSprop/d2/kernel/rmsRMSprop/d2/bias/rmsRMSprop/output/kernel/rmsRMSprop/output/bias/rmsRMSprop/d1/kernel/momentumRMSprop/d1/bias/momentumRMSprop/d2/kernel/momentumRMSprop/d2/bias/momentumRMSprop/output/kernel/momentumRMSprop/output/bias/momentumRMSprop/d1/kernel/mgRMSprop/d1/bias/mgRMSprop/d2/kernel/mgRMSprop/d2/bias/mgRMSprop/output/kernel/mgRMSprop/output/bias/mg*+
Tin$
"2 *
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_21547¨ø
õ
c
*__inference_dropout_d2_layer_call_fn_21280

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20787p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»

"__inference_d2_layer_call_fn_21258

inputs
unknown:
Ô	
	unknown_0:	
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_20709p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÔ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	
 
_user_specified_nameinputs
û	
d
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20827

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_21164

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
£
F
*__inference_dropout_d2_layer_call_fn_21275

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20720a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿

&__inference_output_layer_call_fn_21306

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_20733o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
E

__inference__traced_save_21444
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop(
$savev2_d2_kernel_read_readvariableop&
"savev2_d2_bias_read_readvariableop,
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
.savev2_rmsprop_d2_bias_rms_read_readvariableop8
4savev2_rmsprop_output_kernel_rms_read_readvariableop6
2savev2_rmsprop_output_bias_rms_read_readvariableop9
5savev2_rmsprop_d1_kernel_momentum_read_readvariableop7
3savev2_rmsprop_d1_bias_momentum_read_readvariableop9
5savev2_rmsprop_d2_kernel_momentum_read_readvariableop7
3savev2_rmsprop_d2_bias_momentum_read_readvariableop=
9savev2_rmsprop_output_kernel_momentum_read_readvariableop;
7savev2_rmsprop_output_bias_momentum_read_readvariableop3
/savev2_rmsprop_d1_kernel_mg_read_readvariableop1
-savev2_rmsprop_d1_bias_mg_read_readvariableop3
/savev2_rmsprop_d2_kernel_mg_read_readvariableop1
-savev2_rmsprop_d2_bias_mg_read_readvariableop7
3savev2_rmsprop_output_kernel_mg_read_readvariableop5
1savev2_rmsprop_output_bias_mg_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¶
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ß
valueÕBÒ B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B õ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d2_kernel_read_readvariableop"savev2_d2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_rmsprop_d1_kernel_rms_read_readvariableop.savev2_rmsprop_d1_bias_rms_read_readvariableop0savev2_rmsprop_d2_kernel_rms_read_readvariableop.savev2_rmsprop_d2_bias_rms_read_readvariableop4savev2_rmsprop_output_kernel_rms_read_readvariableop2savev2_rmsprop_output_bias_rms_read_readvariableop5savev2_rmsprop_d1_kernel_momentum_read_readvariableop3savev2_rmsprop_d1_bias_momentum_read_readvariableop5savev2_rmsprop_d2_kernel_momentum_read_readvariableop3savev2_rmsprop_d2_bias_momentum_read_readvariableop9savev2_rmsprop_output_kernel_momentum_read_readvariableop7savev2_rmsprop_output_bias_momentum_read_readvariableop/savev2_rmsprop_d1_kernel_mg_read_readvariableop-savev2_rmsprop_d1_bias_mg_read_readvariableop/savev2_rmsprop_d2_kernel_mg_read_readvariableop-savev2_rmsprop_d2_bias_mg_read_readvariableop3savev2_rmsprop_output_kernel_mg_read_readvariableop1savev2_rmsprop_output_bias_mg_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*
_input_shapesñ
î: :
Ô::
Ô	::	:: : : : : : : :
Ô::
Ô	::	::
Ô::
Ô	::	::
Ô::
Ô	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
Ô:!

_output_shapes	
::&"
 
_output_shapes
:
Ô	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
Ô:!

_output_shapes	
::&"
 
_output_shapes
:
Ô	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
Ô:!

_output_shapes	
::&"
 
_output_shapes
:
Ô	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
Ô:!

_output_shapes	
::&"
 
_output_shapes
:
Ô	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
: 
û	
d
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20787

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

ñ
=__inference_d1_layer_call_and_return_conditional_losses_21209

inputs2
matmul_readvariableop_resource:
Ô.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÔ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
 
_user_specified_nameinputs
æ#
û
@__inference_model_layer_call_and_return_conditional_losses_20969
	mhc_input
peptide_input
d1_20948:
Ô
d1_20950:	
d2_20955:
Ô	
d2_20957:	
output_20961:	
output_20963:
identity¢d1/StatefulPartitionedCall¢d2/StatefulPartitionedCall¢output/StatefulPartitionedCall½
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_20644½
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20652ó
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_20661ï
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_20948d1_20950*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_20675Ù
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20686ñ
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_20695î
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_20955d2_20957*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_20709Ù
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20720
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d2/PartitionedCall:output:0output_20961output_20963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_20733]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namepeptide_input
¼
^
B__inference_flatten_layer_call_and_return_conditional_losses_21175

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
A__inference_output_layer_call_and_return_conditional_losses_21317

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô4
à
@__inference_model_layer_call_and_return_conditional_losses_21153
inputs_0
inputs_15
!d1_matmul_readvariableop_resource:
Ô1
"d1_biasadd_readvariableop_resource:	5
!d2_matmul_readvariableop_resource:
Ô	1
"d2_biasadd_readvariableop_resource:	8
%output_matmul_readvariableop_resource:	4
&output_biasadd_readvariableop_resource:
identity¢d1/BiasAdd/ReadVariableOp¢d1/MatMul/ReadVariableOp¢d2/BiasAdd/ReadVariableOp¢d2/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨  s
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  o
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¨
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
Ô*
dtype0
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_d1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_d1/dropout/MulMuld1/Relu:activations:0!dropout_d1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_d1/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:£
/dropout_d1/dropout/random_uniform/RandomUniformRandomUniform!dropout_d1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_d1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?È
dropout_d1/dropout/GreaterEqualGreaterEqual8dropout_d1/dropout/random_uniform/RandomUniform:output:0*dropout_d1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_d1/dropout/CastCast#dropout_d1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_d1/dropout/Mul_1Muldropout_d1/dropout/Mul:z:0dropout_d1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦
skip1/concatConcatV2concat/concat:output:0dropout_d1/dropout/Mul_1:z:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	|
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
Ô	*
dtype0
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_d2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_d2/dropout/MulMuld2/Relu:activations:0!dropout_d2/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_d2/dropout/ShapeShaped2/Relu:activations:0*
T0*
_output_shapes
:£
/dropout_d2/dropout/random_uniform/RandomUniformRandomUniform!dropout_d2/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_d2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?È
dropout_d2/dropout/GreaterEqualGreaterEqual8dropout_d2/dropout/random_uniform/RandomUniform:output:0*dropout_d2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_d2/dropout/CastCast#dropout_d2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_d2/dropout/Mul_1Muldropout_d2/dropout/Mul:z:0dropout_d2/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
output/MatMulMatMuldropout_d2/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ÿ

ñ
=__inference_d1_layer_call_and_return_conditional_losses_20675

inputs2
matmul_readvariableop_resource:
Ô.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÔ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
 
_user_specified_nameinputs
¾
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_20644

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
¼
^
B__inference_flatten_layer_call_and_return_conditional_losses_20652

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
j
@__inference_skip1_layer_call_and_return_conditional_losses_20695

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
:ÿÿÿÿÿÿÿÿÿÔ	X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÔ:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
c
*__inference_dropout_d1_layer_call_fn_21219

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20827p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ	

%__inference_model_layer_call_fn_21045
inputs_0
inputs_1
unknown:
Ô
	unknown_0:	
	unknown_1:
Ô	
	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20742o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ü
c
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21285

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
m
A__inference_concat_layer_call_and_return_conditional_losses_21188
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
:ÿÿÿÿÿÿÿÿÿÔX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ¨:ÿÿÿÿÿÿÿÿÿ¬:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/1
(

 __inference__wrapped_model_20629
	mhc_input
peptide_input;
'model_d1_matmul_readvariableop_resource:
Ô7
(model_d1_biasadd_readvariableop_resource:	;
'model_d2_matmul_readvariableop_resource:
Ô	7
(model_d2_biasadd_readvariableop_resource:	>
+model_output_matmul_readvariableop_resource:	:
,model_output_biasadd_readvariableop_resource:
identity¢model/d1/BiasAdd/ReadVariableOp¢model/d1/MatMul/ReadVariableOp¢model/d2/BiasAdd/ReadVariableOp¢model/d2/MatMul/ReadVariableOp¢#model/output/BiasAdd/ReadVariableOp¢"model/output/MatMul/ReadVariableOpf
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨  
model/flatten_1/ReshapeReshape	mhc_inputmodel/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  
model/flatten/ReshapeReshapepeptide_inputmodel/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Z
model/concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :À
model/concat/concatConcatV2 model/flatten_1/Reshape:output:0model/flatten/Reshape:output:0!model/concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
model/d1/MatMul/ReadVariableOpReadVariableOp'model_d1_matmul_readvariableop_resource* 
_output_shapes
:
Ô*
dtype0
model/d1/MatMulMatMulmodel/concat/concat:output:0&model/d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/d1/BiasAdd/ReadVariableOpReadVariableOp(model_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/d1/BiasAddBiasAddmodel/d1/MatMul:product:0'model/d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
model/d1/ReluRelumodel/d1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
model/dropout_d1/IdentityIdentitymodel/d1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
model/skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¾
model/skip1/concatConcatV2model/concat/concat:output:0"model/dropout_d1/Identity:output:0 model/skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	
model/d2/MatMul/ReadVariableOpReadVariableOp'model_d2_matmul_readvariableop_resource* 
_output_shapes
:
Ô	*
dtype0
model/d2/MatMulMatMulmodel/skip1/concat:output:0&model/d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/d2/BiasAdd/ReadVariableOpReadVariableOp(model_d2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/d2/BiasAddBiasAddmodel/d2/MatMul:product:0'model/d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
model/d2/ReluRelumodel/d2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
model/dropout_d2/IdentityIdentitymodel/d2/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
model/output/MatMulMatMul"model/dropout_d2/Identity:output:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model/output/SigmoidSigmoidmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitymodel/output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^model/d1/BiasAdd/ReadVariableOp^model/d1/MatMul/ReadVariableOp ^model/d2/BiasAdd/ReadVariableOp^model/d2/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 2B
model/d1/BiasAdd/ReadVariableOpmodel/d1/BiasAdd/ReadVariableOp2@
model/d1/MatMul/ReadVariableOpmodel/d1/MatMul/ReadVariableOp2B
model/d2/BiasAdd/ReadVariableOpmodel/d2/BiasAdd/ReadVariableOp2@
model/d2/MatMul/ReadVariableOpmodel/d2/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namepeptide_input
Ì#
ó
@__inference_model_layer_call_and_return_conditional_losses_20742

inputs
inputs_1
d1_20676:
Ô
d1_20678:	
d2_20710:
Ô	
d2_20712:	
output_20734:	
output_20736:
identity¢d1/StatefulPartitionedCall¢d2/StatefulPartitionedCall¢output/StatefulPartitionedCallº
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_20644¸
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20652ó
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_20661ï
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_20676d1_20678*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_20675Ù
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20686ñ
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_20695î
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_20710d2_20712*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_20709Ù
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20720
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d2/PartitionedCall:output:0output_20734output_20736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_20733]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
d
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21297

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
+
__inference_loss_fn_1_21327
identity]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    X
IdentityIdentity!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
£
F
*__inference_dropout_d1_layer_call_fn_21214

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20686a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
R
&__inference_concat_layer_call_fn_21181
inputs_0
inputs_1
identityº
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_20661a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ¨:ÿÿÿÿÿÿÿÿÿ¬:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/1
Ë~

!__inference__traced_restore_21547
file_prefix.
assignvariableop_d1_kernel:
Ô)
assignvariableop_1_d1_bias:	0
assignvariableop_2_d2_kernel:
Ô	)
assignvariableop_3_d2_bias:	3
 assignvariableop_4_output_kernel:	,
assignvariableop_5_output_bias:)
assignvariableop_6_rmsprop_iter:	 *
 assignvariableop_7_rmsprop_decay: 2
(assignvariableop_8_rmsprop_learning_rate: -
#assignvariableop_9_rmsprop_momentum: )
assignvariableop_10_rmsprop_rho: #
assignvariableop_11_total: #
assignvariableop_12_count: =
)assignvariableop_13_rmsprop_d1_kernel_rms:
Ô6
'assignvariableop_14_rmsprop_d1_bias_rms:	=
)assignvariableop_15_rmsprop_d2_kernel_rms:
Ô	6
'assignvariableop_16_rmsprop_d2_bias_rms:	@
-assignvariableop_17_rmsprop_output_kernel_rms:	9
+assignvariableop_18_rmsprop_output_bias_rms:B
.assignvariableop_19_rmsprop_d1_kernel_momentum:
Ô;
,assignvariableop_20_rmsprop_d1_bias_momentum:	B
.assignvariableop_21_rmsprop_d2_kernel_momentum:
Ô	;
,assignvariableop_22_rmsprop_d2_bias_momentum:	E
2assignvariableop_23_rmsprop_output_kernel_momentum:	>
0assignvariableop_24_rmsprop_output_bias_momentum:<
(assignvariableop_25_rmsprop_d1_kernel_mg:
Ô5
&assignvariableop_26_rmsprop_d1_bias_mg:	<
(assignvariableop_27_rmsprop_d2_kernel_mg:
Ô	5
&assignvariableop_28_rmsprop_d2_bias_mg:	?
,assignvariableop_29_rmsprop_output_kernel_mg:	8
*assignvariableop_30_rmsprop_output_bias_mg:
identity_32¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¹
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*ß
valueÕBÒ B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Á
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_d2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_d2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_rmsprop_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp(assignvariableop_8_rmsprop_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rmsprop_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_rmsprop_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp)assignvariableop_13_rmsprop_d1_kernel_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_rmsprop_d1_bias_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp)assignvariableop_15_rmsprop_d2_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp'assignvariableop_16_rmsprop_d2_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp-assignvariableop_17_rmsprop_output_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp+assignvariableop_18_rmsprop_output_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp.assignvariableop_19_rmsprop_d1_kernel_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp,assignvariableop_20_rmsprop_d1_bias_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp.assignvariableop_21_rmsprop_d2_kernel_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp,assignvariableop_22_rmsprop_d2_bias_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_23AssignVariableOp2assignvariableop_23_rmsprop_output_kernel_momentumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_rmsprop_output_bias_momentumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_rmsprop_d1_kernel_mgIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_rmsprop_d1_bias_mgIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_rmsprop_d2_kernel_mgIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp&assignvariableop_28_rmsprop_d2_bias_mgIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp,assignvariableop_29_rmsprop_output_kernel_mgIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp*assignvariableop_30_rmsprop_output_bias_mgIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ù
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_30AssignVariableOp_302(
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
Ô%
à
@__inference_model_layer_call_and_return_conditional_losses_21101
inputs_0
inputs_15
!d1_matmul_readvariableop_resource:
Ô1
"d1_biasadd_readvariableop_resource:	5
!d2_matmul_readvariableop_resource:
Ô	1
"d2_biasadd_readvariableop_resource:	8
%output_matmul_readvariableop_resource:	4
&output_biasadd_readvariableop_resource:
identity¢d1/BiasAdd/ReadVariableOp¢d1/MatMul/ReadVariableOp¢d2/BiasAdd/ReadVariableOp¢d2/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¨  s
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ,  o
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬T
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¨
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ|
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
Ô*
dtype0
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout_d1/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¦
skip1/concatConcatV2concat/concat:output:0dropout_d1/Identity:output:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	|
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
Ô	*
dtype0
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout_d2/IdentityIdentityd2/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
output/MatMulMatMuldropout_d2/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    a
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
NoOpNoOp^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Â
+
__inference_loss_fn_0_21322
identity]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    X
IdentityIdentity!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
î	

#__inference_signature_wrapper_21025
	mhc_input
peptide_input
unknown:
Ô
	unknown_0:	
	unknown_1:
Ô	
	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_20629o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namepeptide_input
§
E
)__inference_flatten_1_layer_call_fn_21158

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_20644a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ":S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21224

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

ñ
=__inference_d2_layer_call_and_return_conditional_losses_20709

inputs2
matmul_readvariableop_resource:
Ô	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ô	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÔ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	
 
_user_specified_nameinputs
¸
k
A__inference_concat_layer_call_and_return_conditional_losses_20661

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
:ÿÿÿÿÿÿÿÿÿÔX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿ¨:ÿÿÿÿÿÿÿÿÿ¬:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
û	
d
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21236

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs



%__inference_model_layer_call_fn_20941
	mhc_input
peptide_input
unknown:
Ô
	unknown_0:	
	unknown_1:
Ô	
	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20908o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namepeptide_input
¿
l
@__inference_skip1_layer_call_and_return_conditional_losses_21249
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
:ÿÿÿÿÿÿÿÿÿÔ	X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÔ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ÿ

ñ
=__inference_d2_layer_call_and_return_conditional_losses_21270

inputs2
matmul_readvariableop_resource:
Ô	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ô	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÔ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	
 
_user_specified_nameinputs



%__inference_model_layer_call_fn_20757
	mhc_input
peptide_input
unknown:
Ô
	unknown_0:	
	unknown_1:
Ô	
	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20742o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namepeptide_input


ó
A__inference_output_layer_call_and_return_conditional_losses_20733

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã&
Å
@__inference_model_layer_call_and_return_conditional_losses_20997
	mhc_input
peptide_input
d1_20976:
Ô
d1_20978:	
d2_20983:
Ô	
d2_20985:	
output_20989:	
output_20991:
identity¢d1/StatefulPartitionedCall¢d2/StatefulPartitionedCall¢"dropout_d1/StatefulPartitionedCall¢"dropout_d2/StatefulPartitionedCall¢output/StatefulPartitionedCall½
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_20644½
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20652ó
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_20661ï
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_20976d1_20978*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_20675é
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20827ù
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_20695î
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_20983d2_20985*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_20709
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20787
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d2/StatefulPartitionedCall:output:0output_20989output_20991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_20733]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
#
_user_specified_name	mhc_input:ZV
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namepeptide_input
»

"__inference_d1_layer_call_fn_21197

inputs
unknown:
Ô
	unknown_0:	
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_20675p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÔ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
 
_user_specified_nameinputs
É&
½
@__inference_model_layer_call_and_return_conditional_losses_20908

inputs
inputs_1
d1_20887:
Ô
d1_20889:	
d2_20894:
Ô	
d2_20896:	
output_20900:	
output_20902:
identity¢d1/StatefulPartitionedCall¢d2/StatefulPartitionedCall¢"dropout_d1/StatefulPartitionedCall¢"dropout_d2/StatefulPartitionedCall¢output/StatefulPartitionedCallº
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_20644¸
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20652ó
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_20661ï
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0d1_20887d1_20889*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_20675é
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20827ù
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_20695î
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0d2_20894d2_20896*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_20709
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20787
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d2/StatefulPartitionedCall:output:0output_20900output_20902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_20733]
kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    _
kernel/Regularizer_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
NoOpNoOp^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
Q
%__inference_skip1_layer_call_fn_21242
inputs_0
inputs_1
identity¹
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_20695a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÔ:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ü
c
E__inference_dropout_d2_layer_call_and_return_conditional_losses_20720

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ	

%__inference_model_layer_call_fn_21063
inputs_0
inputs_1
unknown:
Ô
	unknown_0:	
	unknown_1:
Ô	
	unknown_2:	
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_20908o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ":ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ü
c
E__inference_dropout_d1_layer_call_and_return_conditional_losses_20686

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
C
'__inference_flatten_layer_call_fn_21169

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_20652a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*þ
serving_defaultê
C
	mhc_input6
serving_default_mhc_input:0ÿÿÿÿÿÿÿÿÿ"
K
peptide_input:
serving_default_peptide_input:0ÿÿÿÿÿÿÿÿÿ:
output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ïó
æ
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
layer_with_weights-2
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
signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
D
#_self_saveable_object_factories"
_tf_keras_input_layer
Ê
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_layer
Ê
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
#%_self_saveable_object_factories"
_tf_keras_layer
Ê
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
#,_self_saveable_object_factories"
_tf_keras_layer
à
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias
#5_self_saveable_object_factories"
_tf_keras_layer
á
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_random_generator
#=_self_saveable_object_factories"
_tf_keras_layer
Ê
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
#D_self_saveable_object_factories"
_tf_keras_layer
à
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
#M_self_saveable_object_factories"
_tf_keras_layer
á
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses
T_random_generator
#U_self_saveable_object_factories"
_tf_keras_layer
à
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
#^_self_saveable_object_factories"
_tf_keras_layer
J
30
41
K2
L3
\4
]5"
trackable_list_wrapper
J
30
41
K2
L3
\4
]5"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
Ê
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ê
ftrace_0
gtrace_1
htrace_2
itrace_32ß
%__inference_model_layer_call_fn_20757
%__inference_model_layer_call_fn_21045
%__inference_model_layer_call_fn_21063
%__inference_model_layer_call_fn_20941À
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
 zftrace_0zgtrace_1zhtrace_2zitrace_3
¶
jtrace_0
ktrace_1
ltrace_2
mtrace_32Ë
@__inference_model_layer_call_and_return_conditional_losses_21101
@__inference_model_layer_call_and_return_conditional_losses_21153
@__inference_model_layer_call_and_return_conditional_losses_20969
@__inference_model_layer_call_and_return_conditional_losses_20997À
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
 zjtrace_0zktrace_1zltrace_2zmtrace_3
ÜBÙ
 __inference__wrapped_model_20629	mhc_inputpeptide_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Â
niter
	odecay
plearning_rate
qmomentum
rrho
3rms¾
4rms¿
KrmsÀ
LrmsÁ
\rmsÂ
]rmsÃ3momentumÄ4momentumÅKmomentumÆLmomentumÇ\momentumÈ]momentumÉ	3mgÊ	4mgË	KmgÌ	LmgÍ	\mgÎ	]mgÏ"
	optimizer
,
sserving_default"
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
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
í
ytrace_02Ð
)__inference_flatten_1_layer_call_fn_21158¢
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
 zytrace_0

ztrace_02ë
D__inference_flatten_1_layer_call_and_return_conditional_losses_21164¢
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
 zztrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_flatten_layer_call_fn_21169¢
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
 ztrace_0

trace_02é
B__inference_flatten_layer_call_and_return_conditional_losses_21175¢
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
 ztrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ì
trace_02Í
&__inference_concat_layer_call_fn_21181¢
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
 ztrace_0

trace_02è
A__inference_concat_layer_call_and_return_conditional_losses_21188¢
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
 ztrace_0
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
'
_0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
è
trace_02É
"__inference_d1_layer_call_fn_21197¢
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
 ztrace_0

trace_02ä
=__inference_d1_layer_call_and_return_conditional_losses_21209¢
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
 ztrace_0
:
Ô2	d1/kernel
:2d1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ê
trace_0
trace_12
*__inference_dropout_d1_layer_call_fn_21214
*__inference_dropout_d1_layer_call_fn_21219´
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
 ztrace_0ztrace_1

trace_0
trace_12Å
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21224
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21236´
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
 ztrace_0ztrace_1
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
ë
trace_02Ì
%__inference_skip1_layer_call_fn_21242¢
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
 ztrace_0

trace_02ç
@__inference_skip1_layer_call_and_return_conditional_losses_21249¢
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
 ztrace_0
 "
trackable_dict_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
²
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
è
¥trace_02É
"__inference_d2_layer_call_fn_21258¢
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
 z¥trace_0

¦trace_02ä
=__inference_d2_layer_call_and_return_conditional_losses_21270¢
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
 z¦trace_0
:
Ô	2	d2/kernel
:2d2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
Ê
¬trace_0
­trace_12
*__inference_dropout_d2_layer_call_fn_21275
*__inference_dropout_d2_layer_call_fn_21280´
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
 z¬trace_0z­trace_1

®trace_0
¯trace_12Å
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21285
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21297´
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
 z®trace_0z¯trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
ì
µtrace_02Í
&__inference_output_layer_call_fn_21306¢
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
 zµtrace_0

¶trace_02è
A__inference_output_layer_call_and_return_conditional_losses_21317¢
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
 z¶trace_0
 :	2output/kernel
:2output/bias
 "
trackable_dict_wrapper
Î
·trace_02¯
__inference_loss_fn_0_21322
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
annotationsª *¢ z·trace_0
Î
¸trace_02¯
__inference_loss_fn_1_21327
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
annotationsª *¢ z¸trace_0
 "
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
(
¹0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
%__inference_model_layer_call_fn_20757	mhc_inputpeptide_input"À
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
B
%__inference_model_layer_call_fn_21045inputs/0inputs/1"À
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
B
%__inference_model_layer_call_fn_21063inputs/0inputs/1"À
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
B
%__inference_model_layer_call_fn_20941	mhc_inputpeptide_input"À
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
B
@__inference_model_layer_call_and_return_conditional_losses_21101inputs/0inputs/1"À
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
B
@__inference_model_layer_call_and_return_conditional_losses_21153inputs/0inputs/1"À
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
¤B¡
@__inference_model_layer_call_and_return_conditional_losses_20969	mhc_inputpeptide_input"À
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
¤B¡
@__inference_model_layer_call_and_return_conditional_losses_20997	mhc_inputpeptide_input"À
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
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
ÙBÖ
#__inference_signature_wrapper_21025	mhc_inputpeptide_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÝBÚ
)__inference_flatten_1_layer_call_fn_21158inputs"¢
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
øBõ
D__inference_flatten_1_layer_call_and_return_conditional_losses_21164inputs"¢
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
ÛBØ
'__inference_flatten_layer_call_fn_21169inputs"¢
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
öBó
B__inference_flatten_layer_call_and_return_conditional_losses_21175inputs"¢
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
æBã
&__inference_concat_layer_call_fn_21181inputs/0inputs/1"¢
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
Bþ
A__inference_concat_layer_call_and_return_conditional_losses_21188inputs/0inputs/1"¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
_0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÖBÓ
"__inference_d1_layer_call_fn_21197inputs"¢
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
ñBî
=__inference_d1_layer_call_and_return_conditional_losses_21209inputs"¢
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
ðBí
*__inference_dropout_d1_layer_call_fn_21214inputs"´
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
ðBí
*__inference_dropout_d1_layer_call_fn_21219inputs"´
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
B
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21224inputs"´
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
B
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21236inputs"´
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
åBâ
%__inference_skip1_layer_call_fn_21242inputs/0inputs/1"¢
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
Bý
@__inference_skip1_layer_call_and_return_conditional_losses_21249inputs/0inputs/1"¢
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÖBÓ
"__inference_d2_layer_call_fn_21258inputs"¢
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
ñBî
=__inference_d2_layer_call_and_return_conditional_losses_21270inputs"¢
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
ðBí
*__inference_dropout_d2_layer_call_fn_21275inputs"´
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
ðBí
*__inference_dropout_d2_layer_call_fn_21280inputs"´
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
B
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21285inputs"´
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
B
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21297inputs"´
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
ÚB×
&__inference_output_layer_call_fn_21306inputs"¢
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
õBò
A__inference_output_layer_call_and_return_conditional_losses_21317inputs"¢
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
²B¯
__inference_loss_fn_0_21322"
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
²B¯
__inference_loss_fn_1_21327"
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
R
º	variables
»	keras_api

¼total

½count"
_tf_keras_metric
0
¼0
½1"
trackable_list_wrapper
.
º	variables"
_generic_user_object
:  (2total
:  (2count
':%
Ô2RMSprop/d1/kernel/rms
 :2RMSprop/d1/bias/rms
':%
Ô	2RMSprop/d2/kernel/rms
 :2RMSprop/d2/bias/rms
*:(	2RMSprop/output/kernel/rms
#:!2RMSprop/output/bias/rms
,:*
Ô2RMSprop/d1/kernel/momentum
%:#2RMSprop/d1/bias/momentum
,:*
Ô	2RMSprop/d2/kernel/momentum
%:#2RMSprop/d2/bias/momentum
/:-	2RMSprop/output/kernel/momentum
(:&2RMSprop/output/bias/momentum
&:$
Ô2RMSprop/d1/kernel/mg
:2RMSprop/d1/bias/mg
&:$
Ô	2RMSprop/d2/kernel/mg
:2RMSprop/d2/bias/mg
):'	2RMSprop/output/kernel/mg
": 2RMSprop/output/bias/mgÈ
 __inference__wrapped_model_20629£34KL\]h¢e
^¢[
YV
'$
	mhc_inputÿÿÿÿÿÿÿÿÿ"
+(
peptide_inputÿÿÿÿÿÿÿÿÿ
ª "/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿÌ
A__inference_concat_layer_call_and_return_conditional_losses_21188\¢Y
R¢O
MJ
# 
inputs/0ÿÿÿÿÿÿÿÿÿ¨
# 
inputs/1ÿÿÿÿÿÿÿÿÿ¬
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÔ
 £
&__inference_concat_layer_call_fn_21181y\¢Y
R¢O
MJ
# 
inputs/0ÿÿÿÿÿÿÿÿÿ¨
# 
inputs/1ÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿÔ
=__inference_d1_layer_call_and_return_conditional_losses_21209^340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÔ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 w
"__inference_d1_layer_call_fn_21197Q340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÔ
ª "ÿÿÿÿÿÿÿÿÿ
=__inference_d2_layer_call_and_return_conditional_losses_21270^KL0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÔ	
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 w
"__inference_d2_layer_call_fn_21258QKL0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÔ	
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21224^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_d1_layer_call_and_return_conditional_losses_21236^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_d1_layer_call_fn_21214Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_d1_layer_call_fn_21219Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21285^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_d2_layer_call_and_return_conditional_losses_21297^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_d2_layer_call_fn_21275Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_d2_layer_call_fn_21280Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_flatten_1_layer_call_and_return_conditional_losses_21164]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ"
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¨
 }
)__inference_flatten_1_layer_call_fn_21158P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ"
ª "ÿÿÿÿÿÿÿÿÿ¨£
B__inference_flatten_layer_call_and_return_conditional_losses_21175]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 {
'__inference_flatten_layer_call_fn_21169P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬7
__inference_loss_fn_0_21322¢

¢ 
ª " 7
__inference_loss_fn_1_21327¢

¢ 
ª " æ
@__inference_model_layer_call_and_return_conditional_losses_20969¡34KL\]p¢m
f¢c
YV
'$
	mhc_inputÿÿÿÿÿÿÿÿÿ"
+(
peptide_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 æ
@__inference_model_layer_call_and_return_conditional_losses_20997¡34KL\]p¢m
f¢c
YV
'$
	mhc_inputÿÿÿÿÿÿÿÿÿ"
+(
peptide_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 à
@__inference_model_layer_call_and_return_conditional_losses_2110134KL\]j¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ"
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 à
@__inference_model_layer_call_and_return_conditional_losses_2115334KL\]j¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ"
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
%__inference_model_layer_call_fn_2075734KL\]p¢m
f¢c
YV
'$
	mhc_inputÿÿÿÿÿÿÿÿÿ"
+(
peptide_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¾
%__inference_model_layer_call_fn_2094134KL\]p¢m
f¢c
YV
'$
	mhc_inputÿÿÿÿÿÿÿÿÿ"
+(
peptide_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¸
%__inference_model_layer_call_fn_2104534KL\]j¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ"
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¸
%__inference_model_layer_call_fn_2106334KL\]j¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ"
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¢
A__inference_output_layer_call_and_return_conditional_losses_21317]\]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
&__inference_output_layer_call_fn_21306P\]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿå
#__inference_signature_wrapper_21025½34KL\]¢~
¢ 
wªt
4
	mhc_input'$
	mhc_inputÿÿÿÿÿÿÿÿÿ"
<
peptide_input+(
peptide_inputÿÿÿÿÿÿÿÿÿ"/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿË
@__inference_skip1_layer_call_and_return_conditional_losses_21249\¢Y
R¢O
MJ
# 
inputs/0ÿÿÿÿÿÿÿÿÿÔ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÔ	
 ¢
%__inference_skip1_layer_call_fn_21242y\¢Y
R¢O
MJ
# 
inputs/0ÿÿÿÿÿÿÿÿÿÔ
# 
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÔ	