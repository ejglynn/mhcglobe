��
��
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-6-g2dd7e988��

p
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name	d1/kernel
i
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel* 
_output_shapes
:
��*
dtype0
g
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d1/bias
`
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes	
:�*
dtype0
p
	d2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*
shared_name	d2/kernel
i
d2/kernel/Read/ReadVariableOpReadVariableOp	d2/kernel* 
_output_shapes
:
�	�*
dtype0
g
d2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d2/bias
`
d2/bias/Read/ReadVariableOpReadVariableOpd2/bias*
_output_shapes	
:�*
dtype0
p
	d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name	d3/kernel
i
d3/kernel/Read/ReadVariableOpReadVariableOp	d3/kernel* 
_output_shapes
:
��*
dtype0
g
d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	d3/bias
`
d3/bias/Read/ReadVariableOpReadVariableOpd3/bias*
_output_shapes	
:�*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	�*
dtype0
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
�
RMSprop/d1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameRMSprop/d1/kernel/rms
�
)RMSprop/d1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/rms* 
_output_shapes
:
��*
dtype0

RMSprop/d1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameRMSprop/d1/bias/rms
x
'RMSprop/d1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/rms*
_output_shapes	
:�*
dtype0
�
RMSprop/d2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*&
shared_nameRMSprop/d2/kernel/rms
�
)RMSprop/d2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/rms* 
_output_shapes
:
�	�*
dtype0

RMSprop/d2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameRMSprop/d2/bias/rms
x
'RMSprop/d2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/rms*
_output_shapes	
:�*
dtype0
�
RMSprop/d3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameRMSprop/d3/kernel/rms
�
)RMSprop/d3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d3/kernel/rms* 
_output_shapes
:
��*
dtype0

RMSprop/d3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameRMSprop/d3/bias/rms
x
'RMSprop/d3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d3/bias/rms*
_output_shapes	
:�*
dtype0
�
RMSprop/output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�**
shared_nameRMSprop/output/kernel/rms
�
-RMSprop/output/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/rms*
_output_shapes
:	�*
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
RMSprop/d1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*+
shared_nameRMSprop/d1/kernel/momentum
�
.RMSprop/d1/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
RMSprop/d1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameRMSprop/d1/bias/momentum
�
,RMSprop/d1/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/momentum*
_output_shapes	
:�*
dtype0
�
RMSprop/d2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*+
shared_nameRMSprop/d2/kernel/momentum
�
.RMSprop/d2/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/momentum* 
_output_shapes
:
�	�*
dtype0
�
RMSprop/d2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameRMSprop/d2/bias/momentum
�
,RMSprop/d2/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/momentum*
_output_shapes	
:�*
dtype0
�
RMSprop/d3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*+
shared_nameRMSprop/d3/kernel/momentum
�
.RMSprop/d3/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d3/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
RMSprop/d3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameRMSprop/d3/bias/momentum
�
,RMSprop/d3/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d3/bias/momentum*
_output_shapes	
:�*
dtype0
�
RMSprop/output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name RMSprop/output/kernel/momentum
�
2RMSprop/output/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/momentum*
_output_shapes
:	�*
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
RMSprop/d1/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameRMSprop/d1/kernel/mg

(RMSprop/d1/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/mg* 
_output_shapes
:
��*
dtype0
}
RMSprop/d1/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameRMSprop/d1/bias/mg
v
&RMSprop/d1/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/mg*
_output_shapes	
:�*
dtype0
�
RMSprop/d2/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�	�*%
shared_nameRMSprop/d2/kernel/mg

(RMSprop/d2/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d2/kernel/mg* 
_output_shapes
:
�	�*
dtype0
}
RMSprop/d2/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameRMSprop/d2/bias/mg
v
&RMSprop/d2/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d2/bias/mg*
_output_shapes	
:�*
dtype0
�
RMSprop/d3/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameRMSprop/d3/kernel/mg

(RMSprop/d3/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d3/kernel/mg* 
_output_shapes
:
��*
dtype0
}
RMSprop/d3/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameRMSprop/d3/bias/mg
v
&RMSprop/d3/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d3/bias/mg*
_output_shapes	
:�*
dtype0
�
RMSprop/output/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameRMSprop/output/kernel/mg
�
,RMSprop/output/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/mg*
_output_shapes
:	�*
dtype0
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

NoOpNoOp
�B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�A
value�AB�A B�A
�
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
layer-11
layer_with_weights-3
layer-12
	optimizer

signatures
	variables
trainable_variables
regularization_losses
	keras_api
 
 
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
R
4	variables
5trainable_variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
�
Hiter
	Idecay
Jlearning_rate
Kmomentum
Lrho	 rms}	!rms~	.rms
/rms�
8rms�
9rms�
Brms�
Crms� momentum�!momentum�.momentum�/momentum�8momentum�9momentum�Bmomentum�Cmomentum�	 mg�	!mg�	.mg�	/mg�	8mg�	9mg�	Bmg�	Cmg�
 
8
 0
!1
.2
/3
84
95
B6
C7
8
 0
!1
.2
/3
84
95
B6
C7
 
�
Mlayer_regularization_losses
	variables
Nmetrics

Olayers
trainable_variables
Pnon_trainable_variables
regularization_losses
 
 
 
�
Qlayer_regularization_losses
	variables
Rmetrics

Slayers
trainable_variables
Tnon_trainable_variables
regularization_losses
 
 
 
�
Ulayer_regularization_losses
	variables
Vmetrics

Wlayers
trainable_variables
Xnon_trainable_variables
regularization_losses
 
 
 
�
Ylayer_regularization_losses
	variables
Zmetrics

[layers
trainable_variables
\non_trainable_variables
regularization_losses
US
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
�
]layer_regularization_losses
"	variables
^metrics

_layers
#trainable_variables
`non_trainable_variables
$regularization_losses
 
 
 
�
alayer_regularization_losses
&	variables
bmetrics

clayers
'trainable_variables
dnon_trainable_variables
(regularization_losses
 
 
 
�
elayer_regularization_losses
*	variables
fmetrics

glayers
+trainable_variables
hnon_trainable_variables
,regularization_losses
US
VARIABLE_VALUE	d2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
�
ilayer_regularization_losses
0	variables
jmetrics

klayers
1trainable_variables
lnon_trainable_variables
2regularization_losses
 
 
 
�
mlayer_regularization_losses
4	variables
nmetrics

olayers
5trainable_variables
pnon_trainable_variables
6regularization_losses
US
VARIABLE_VALUE	d3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
�
qlayer_regularization_losses
:	variables
rmetrics

slayers
;trainable_variables
tnon_trainable_variables
<regularization_losses
 
 
 
�
ulayer_regularization_losses
>	variables
vmetrics

wlayers
?trainable_variables
xnon_trainable_variables
@regularization_losses
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
�
ylayer_regularization_losses
D	variables
zmetrics

{layers
Etrainable_variables
|non_trainable_variables
Fregularization_losses
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
 
^
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
 
 
 
 
}
VARIABLE_VALUERMSprop/d1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUERMSprop/d1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/d2/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUERMSprop/d2/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/d3/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUERMSprop/d3/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/output/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/output/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/d1/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/d1/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/d2/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/d2/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/d3/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/d3/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/output/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/output/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/d1/kernel/mgSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUERMSprop/d1/bias/mgQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/d2/kernel/mgSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUERMSprop/d2/bias/mgQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/d3/kernel/mgSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUERMSprop/d3/bias/mgQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUERMSprop/output/kernel/mgSlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/output/bias/mgQlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
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
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_486459
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd2/kernel/Read/ReadVariableOpd2/bias/Read/ReadVariableOpd3/kernel/Read/ReadVariableOpd3/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp)RMSprop/d1/kernel/rms/Read/ReadVariableOp'RMSprop/d1/bias/rms/Read/ReadVariableOp)RMSprop/d2/kernel/rms/Read/ReadVariableOp'RMSprop/d2/bias/rms/Read/ReadVariableOp)RMSprop/d3/kernel/rms/Read/ReadVariableOp'RMSprop/d3/bias/rms/Read/ReadVariableOp-RMSprop/output/kernel/rms/Read/ReadVariableOp+RMSprop/output/bias/rms/Read/ReadVariableOp.RMSprop/d1/kernel/momentum/Read/ReadVariableOp,RMSprop/d1/bias/momentum/Read/ReadVariableOp.RMSprop/d2/kernel/momentum/Read/ReadVariableOp,RMSprop/d2/bias/momentum/Read/ReadVariableOp.RMSprop/d3/kernel/momentum/Read/ReadVariableOp,RMSprop/d3/bias/momentum/Read/ReadVariableOp2RMSprop/output/kernel/momentum/Read/ReadVariableOp0RMSprop/output/bias/momentum/Read/ReadVariableOp(RMSprop/d1/kernel/mg/Read/ReadVariableOp&RMSprop/d1/bias/mg/Read/ReadVariableOp(RMSprop/d2/kernel/mg/Read/ReadVariableOp&RMSprop/d2/bias/mg/Read/ReadVariableOp(RMSprop/d3/kernel/mg/Read/ReadVariableOp&RMSprop/d3/bias/mg/Read/ReadVariableOp,RMSprop/output/kernel/mg/Read/ReadVariableOp*RMSprop/output/bias/mg/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_486607
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d2/kerneld2/bias	d3/kerneld3/biasoutput/kerneloutput/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoRMSprop/d1/kernel/rmsRMSprop/d1/bias/rmsRMSprop/d2/kernel/rmsRMSprop/d2/bias/rmsRMSprop/d3/kernel/rmsRMSprop/d3/bias/rmsRMSprop/output/kernel/rmsRMSprop/output/bias/rmsRMSprop/d1/kernel/momentumRMSprop/d1/bias/momentumRMSprop/d2/kernel/momentumRMSprop/d2/bias/momentumRMSprop/d3/kernel/momentumRMSprop/d3/bias/momentumRMSprop/output/kernel/momentumRMSprop/output/bias/momentumRMSprop/d1/kernel/mgRMSprop/d1/bias/mgRMSprop/d2/kernel/mgRMSprop/d2/bias/mgRMSprop/d3/kernel/mgRMSprop/d3/bias/mgRMSprop/output/kernel/mgRMSprop/output/bias/mg*1
Tin*
(2&*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_486730��
�

�
=__inference_d1_layer_call_and_return_conditional_losses_73915

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
+
__inference_loss_fn_0_73664
identity
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Constg
IdentityIdentity$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
c
*__inference_dropout_d2_layer_call_fn_73891

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_738862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
A__inference_output_layer_call_and_return_conditional_losses_74021

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
F
*__inference_dropout_d3_layer_call_fn_73641

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_736362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_74115
	mhc_input
peptide_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_741012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
�	
�
=__inference_d3_layer_call_and_return_conditional_losses_73834

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
D
(__inference_restored_function_body_74507

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_734242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
c
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73636

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
R
&__inference_concat_layer_call_fn_73443
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_734372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
c
*__inference_dropout_d1_layer_call_fn_73823

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_738182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_74075
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_740472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
!
cond_false_486547
identityz
ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2480510458f648b087ba6f4e8c54c641/part2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
�
%__inference_model_layer_call_fn_74061
	mhc_input
peptide_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_740472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
�
�
%__inference_model_layer_call_fn_74129
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_741012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
;__forward_d1_layer_call_and_return_conditional_losses_75331
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
relu
matmul_readvariableop

inputs��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
reluRelu:activations:0*/
_input_shapes
:����������::*j
backward_function_namePN__inference___backward_d1_layer_call_and_return_conditional_losses_75317_7533220
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_restored_function_body_74449

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_739032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������	::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73424

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
 
cond_true_486546
identityT
ConstConst*
_output_shapes
: *
dtype0*
valueB B.part2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
+
__inference_loss_fn_1_73673
identity
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Constg
IdentityIdentity$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
�
D
(__inference_restored_function_body_74385

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_736602
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
E
)__inference_flatten_1_layer_call_fn_74010

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_740052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������":& "
 
_user_specified_nameinputs
�
c
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73793

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
d
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73591

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�,
�
@__inference_model_layer_call_and_return_conditional_losses_74101

inputs
inputs_1%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2%
!d3_statefulpartitionedcall_args_1%
!d3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_740052
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_733872
flatten/PartitionedCall�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_734372
concat/PartitionedCall�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_739152
d1/StatefulPartitionedCall�
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_737242
dropout_d1/PartitionedCall�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������	*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_736482
skip1/PartitionedCall�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_739342
d2/StatefulPartitionedCall�
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_737932
dropout_d2/PartitionedCall�
d3/StatefulPartitionedCallStatefulPartitionedCall#dropout_d2/PartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_737812
d3/StatefulPartitionedCall�
dropout_d3/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_736362
dropout_d3/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d3/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_740212 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�

�
=__inference_d2_layer_call_and_return_conditional_losses_73903

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
;__forward_d2_layer_call_and_return_conditional_losses_75198
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
relu
matmul_readvariableop

inputs��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
reluRelu:activations:0*/
_input_shapes
:����������	::*j
backward_function_namePN__inference___backward_d2_layer_call_and_return_conditional_losses_75184_7519920
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
d
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73818

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�1
�
@__inference_model_layer_call_and_return_conditional_losses_74047

inputs
inputs_1%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2%
!d3_statefulpartitionedcall_args_1%
!d3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�"dropout_d1/StatefulPartitionedCall�"dropout_d2/StatefulPartitionedCall�"dropout_d3/StatefulPartitionedCall�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_740052
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_733872
flatten/PartitionedCall�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_734372
concat/PartitionedCall�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_739152
d1/StatefulPartitionedCall�
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_738182$
"dropout_d1/StatefulPartitionedCall�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������	*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_736482
skip1/PartitionedCall�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_739342
d2/StatefulPartitionedCall�
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_738862$
"dropout_d2/StatefulPartitionedCall�
d3/StatefulPartitionedCallStatefulPartitionedCall+dropout_d2/StatefulPartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_737812
d3/StatefulPartitionedCall�
"dropout_d3/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0#^dropout_d2/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_736112$
"dropout_d3/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d3/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_740212 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall#^dropout_d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2H
"dropout_d3/StatefulPartitionedCall"dropout_d3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
d
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73886

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
m
A__inference_concat_layer_call_and_return_conditional_losses_73736
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
c
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73669

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
D
(__inference_restored_function_body_74377

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_734302
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������":& "
 
_user_specified_nameinputs
�
�
?__forward_output_layer_call_and_return_conditional_losses_75014
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
sigmoid
matmul_readvariableop

inputs��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
sigmoidSigmoid:y:0*/
_input_shapes
:����������::*n
backward_function_nameTR__inference___backward_output_layer_call_and_return_conditional_losses_75000_7501520
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�m
�
@__inference_model_layer_call_and_return_conditional_losses_73559
inputs_0
inputs_1%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource%
!d2_matmul_readvariableop_resource&
"d2_biasadd_readvariableop_resource%
!d3_matmul_readvariableop_resource&
"d3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��d1/BiasAdd/ReadVariableOp�d1/MatMul/ReadVariableOp�d2/BiasAdd/ReadVariableOp�d2/MatMul/ReadVariableOp�d3/BiasAdd/ReadVariableOp�d3/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_1/Const�
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  2
flatten/Const�
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshapej
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat/axis�
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concat/concat�
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
d1/MatMul/ReadVariableOp�
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
	d1/MatMul�
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
d1/BiasAdd/ReadVariableOp�

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2	
d1/Reluw
dropout_d1/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_d1/dropout/ratey
dropout_d1/dropout/ShapeShaped1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_d1/dropout/Shape�
%dropout_d1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d1/dropout/random_uniform/min�
%dropout_d1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%dropout_d1/dropout/random_uniform/max�
/dropout_d1/dropout/random_uniform/RandomUniformRandomUniform!dropout_d1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_d1/dropout/random_uniform/RandomUniform�
%dropout_d1/dropout/random_uniform/subSub.dropout_d1/dropout/random_uniform/max:output:0.dropout_d1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d1/dropout/random_uniform/sub�
%dropout_d1/dropout/random_uniform/mulMul8dropout_d1/dropout/random_uniform/RandomUniform:output:0)dropout_d1/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2'
%dropout_d1/dropout/random_uniform/mul�
!dropout_d1/dropout/random_uniformAdd)dropout_d1/dropout/random_uniform/mul:z:0.dropout_d1/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2#
!dropout_d1/dropout/random_uniformy
dropout_d1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_d1/dropout/sub/x�
dropout_d1/dropout/subSub!dropout_d1/dropout/sub/x:output:0 dropout_d1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d1/dropout/sub�
dropout_d1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_d1/dropout/truediv/x�
dropout_d1/dropout/truedivRealDiv%dropout_d1/dropout/truediv/x:output:0dropout_d1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d1/dropout/truediv�
dropout_d1/dropout/GreaterEqualGreaterEqual%dropout_d1/dropout/random_uniform:z:0 dropout_d1/dropout/rate:output:0*
T0*(
_output_shapes
:����������2!
dropout_d1/dropout/GreaterEqual�
dropout_d1/dropout/mulMuld1/Relu:activations:0dropout_d1/dropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout_d1/dropout/mul�
dropout_d1/dropout/CastCast#dropout_d1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_d1/dropout/Cast�
dropout_d1/dropout/mul_1Muldropout_d1/dropout/mul:z:0dropout_d1/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_d1/dropout/mul_1h
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip1/concat/axis�
skip1/concatConcatV2concat/concat:output:0dropout_d1/dropout/mul_1:z:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	2
skip1/concat�
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
d2/MatMul/ReadVariableOp�
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
	d2/MatMul�
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
d2/BiasAdd/ReadVariableOp�

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2	
d2/Reluw
dropout_d2/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_d2/dropout/ratey
dropout_d2/dropout/ShapeShaped2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_d2/dropout/Shape�
%dropout_d2/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d2/dropout/random_uniform/min�
%dropout_d2/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%dropout_d2/dropout/random_uniform/max�
/dropout_d2/dropout/random_uniform/RandomUniformRandomUniform!dropout_d2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_d2/dropout/random_uniform/RandomUniform�
%dropout_d2/dropout/random_uniform/subSub.dropout_d2/dropout/random_uniform/max:output:0.dropout_d2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d2/dropout/random_uniform/sub�
%dropout_d2/dropout/random_uniform/mulMul8dropout_d2/dropout/random_uniform/RandomUniform:output:0)dropout_d2/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2'
%dropout_d2/dropout/random_uniform/mul�
!dropout_d2/dropout/random_uniformAdd)dropout_d2/dropout/random_uniform/mul:z:0.dropout_d2/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2#
!dropout_d2/dropout/random_uniformy
dropout_d2/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_d2/dropout/sub/x�
dropout_d2/dropout/subSub!dropout_d2/dropout/sub/x:output:0 dropout_d2/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d2/dropout/sub�
dropout_d2/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_d2/dropout/truediv/x�
dropout_d2/dropout/truedivRealDiv%dropout_d2/dropout/truediv/x:output:0dropout_d2/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d2/dropout/truediv�
dropout_d2/dropout/GreaterEqualGreaterEqual%dropout_d2/dropout/random_uniform:z:0 dropout_d2/dropout/rate:output:0*
T0*(
_output_shapes
:����������2!
dropout_d2/dropout/GreaterEqual�
dropout_d2/dropout/mulMuld2/Relu:activations:0dropout_d2/dropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout_d2/dropout/mul�
dropout_d2/dropout/CastCast#dropout_d2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_d2/dropout/Cast�
dropout_d2/dropout/mul_1Muldropout_d2/dropout/mul:z:0dropout_d2/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_d2/dropout/mul_1�
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
d3/MatMul/ReadVariableOp�
	d3/MatMulMatMuldropout_d2/dropout/mul_1:z:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
	d3/MatMul�
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
d3/BiasAdd/ReadVariableOp�

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

d3/BiasAddb
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2	
d3/Reluw
dropout_d3/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_d3/dropout/ratey
dropout_d3/dropout/ShapeShaped3/Relu:activations:0*
T0*
_output_shapes
:2
dropout_d3/dropout/Shape�
%dropout_d3/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d3/dropout/random_uniform/min�
%dropout_d3/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2'
%dropout_d3/dropout/random_uniform/max�
/dropout_d3/dropout/random_uniform/RandomUniformRandomUniform!dropout_d3/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_d3/dropout/random_uniform/RandomUniform�
%dropout_d3/dropout/random_uniform/subSub.dropout_d3/dropout/random_uniform/max:output:0.dropout_d3/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d3/dropout/random_uniform/sub�
%dropout_d3/dropout/random_uniform/mulMul8dropout_d3/dropout/random_uniform/RandomUniform:output:0)dropout_d3/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2'
%dropout_d3/dropout/random_uniform/mul�
!dropout_d3/dropout/random_uniformAdd)dropout_d3/dropout/random_uniform/mul:z:0.dropout_d3/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2#
!dropout_d3/dropout/random_uniformy
dropout_d3/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_d3/dropout/sub/x�
dropout_d3/dropout/subSub!dropout_d3/dropout/sub/x:output:0 dropout_d3/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d3/dropout/sub�
dropout_d3/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_d3/dropout/truediv/x�
dropout_d3/dropout/truedivRealDiv%dropout_d3/dropout/truediv/x:output:0dropout_d3/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d3/dropout/truediv�
dropout_d3/dropout/GreaterEqualGreaterEqual%dropout_d3/dropout/random_uniform:z:0 dropout_d3/dropout/rate:output:0*
T0*(
_output_shapes
:����������2!
dropout_d3/dropout/GreaterEqual�
dropout_d3/dropout/mulMuld3/Relu:activations:0dropout_d3/dropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout_d3/dropout/mul�
dropout_d3/dropout/CastCast#dropout_d3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_d3/dropout/Cast�
dropout_d3/dropout/mul_1Muldropout_d3/dropout/mul:z:0dropout_d3/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_d3/dropout/mul_1�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_d3/dropout/mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAddv
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output/Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentityoutput/Sigmoid:y:0^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
k
A__inference_concat_layer_call_and_return_conditional_losses_73437

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
(__inference_restored_function_body_74485

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_738342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_73392

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_733872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
F
*__inference_dropout_d1_layer_call_fn_73729

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_737242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
"__inference_d2_layer_call_fn_73941

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_739342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������	::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
"__inference_d1_layer_call_fn_73922

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_739152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
*__inference_dropout_d3_layer_call_fn_73616

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_736112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73468

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
d
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73611

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�	
�
A__inference_output_layer_call_and_return_conditional_losses_73631

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_restored_function_body_74404

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_735712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_73660

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_74005

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������":& "
 
_user_specified_nameinputs
�N
�
__inference__traced_save_486607
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
&savev2_rmsprop_rho_read_readvariableop4
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
1savev2_rmsprop_output_bias_mg_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:0*
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatch�
condStatelessIfStaticRegexFullMatch:output:0"/device:CPU:0*
Tcond0
*	
Tin
 *
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *$
else_branchR
cond_false_486547*
output_shapes
: *#
then_branchR
cond_true_4865462
condi
cond/IdentityIdentitycond:output:0"/device:CPU:0*
T0*
_output_shapes
: 2
cond/Identity{

StringJoin
StringJoinfile_prefixcond/Identity:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d2_kernel_read_readvariableop"savev2_d2_bias_read_readvariableop$savev2_d3_kernel_read_readvariableop"savev2_d3_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop0savev2_rmsprop_d1_kernel_rms_read_readvariableop.savev2_rmsprop_d1_bias_rms_read_readvariableop0savev2_rmsprop_d2_kernel_rms_read_readvariableop.savev2_rmsprop_d2_bias_rms_read_readvariableop0savev2_rmsprop_d3_kernel_rms_read_readvariableop.savev2_rmsprop_d3_bias_rms_read_readvariableop4savev2_rmsprop_output_kernel_rms_read_readvariableop2savev2_rmsprop_output_bias_rms_read_readvariableop5savev2_rmsprop_d1_kernel_momentum_read_readvariableop3savev2_rmsprop_d1_bias_momentum_read_readvariableop5savev2_rmsprop_d2_kernel_momentum_read_readvariableop3savev2_rmsprop_d2_bias_momentum_read_readvariableop5savev2_rmsprop_d3_kernel_momentum_read_readvariableop3savev2_rmsprop_d3_bias_momentum_read_readvariableop9savev2_rmsprop_output_kernel_momentum_read_readvariableop7savev2_rmsprop_output_bias_momentum_read_readvariableop/savev2_rmsprop_d1_kernel_mg_read_readvariableop-savev2_rmsprop_d1_bias_mg_read_readvariableop/savev2_rmsprop_d2_kernel_mg_read_readvariableop-savev2_rmsprop_d2_bias_mg_read_readvariableop/savev2_rmsprop_d3_kernel_mg_read_readvariableop-savev2_rmsprop_d3_bias_mg_read_readvariableop3savev2_rmsprop_output_kernel_mg_read_readvariableop1savev2_rmsprop_output_bias_mg_read_readvariableop"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
�	�:�:
��:�:	�:: : : : : :
��:�:
�	�:�:
��:�:	�::
��:�:
�	�:�:
��:�:	�::
��:�:
�	�:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
D
(__inference_restored_function_body_74471

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_736692
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
D
(__inference_restored_function_body_74426

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_734682
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
Q
%__inference_skip1_layer_call_fn_73654
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������	*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_736482
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
��
�
"__inference__traced_restore_486730
file_prefix
assignvariableop_d1_kernel
assignvariableop_1_d1_bias 
assignvariableop_2_d2_kernel
assignvariableop_3_d2_bias 
assignvariableop_4_d3_kernel
assignvariableop_5_d3_bias$
 assignvariableop_6_output_kernel"
assignvariableop_7_output_bias#
assignvariableop_8_rmsprop_iter$
 assignvariableop_9_rmsprop_decay-
)assignvariableop_10_rmsprop_learning_rate(
$assignvariableop_11_rmsprop_momentum#
assignvariableop_12_rmsprop_rho-
)assignvariableop_13_rmsprop_d1_kernel_rms+
'assignvariableop_14_rmsprop_d1_bias_rms-
)assignvariableop_15_rmsprop_d2_kernel_rms+
'assignvariableop_16_rmsprop_d2_bias_rms-
)assignvariableop_17_rmsprop_d3_kernel_rms+
'assignvariableop_18_rmsprop_d3_bias_rms1
-assignvariableop_19_rmsprop_output_kernel_rms/
+assignvariableop_20_rmsprop_output_bias_rms2
.assignvariableop_21_rmsprop_d1_kernel_momentum0
,assignvariableop_22_rmsprop_d1_bias_momentum2
.assignvariableop_23_rmsprop_d2_kernel_momentum0
,assignvariableop_24_rmsprop_d2_bias_momentum2
.assignvariableop_25_rmsprop_d3_kernel_momentum0
,assignvariableop_26_rmsprop_d3_bias_momentum6
2assignvariableop_27_rmsprop_output_kernel_momentum4
0assignvariableop_28_rmsprop_output_bias_momentum,
(assignvariableop_29_rmsprop_d1_kernel_mg*
&assignvariableop_30_rmsprop_d1_bias_mg,
(assignvariableop_31_rmsprop_d2_kernel_mg*
&assignvariableop_32_rmsprop_d2_bias_mg,
(assignvariableop_33_rmsprop_d3_kernel_mg*
&assignvariableop_34_rmsprop_d3_bias_mg0
,assignvariableop_35_rmsprop_output_kernel_mg.
*assignvariableop_36_rmsprop_output_bias_mg
identity_38��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_d1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_d1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_d2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_d2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_d3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_d3_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_output_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_output_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_rmsprop_rhoIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_rmsprop_d1_kernel_rmsIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_rmsprop_d1_bias_rmsIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_rmsprop_d2_kernel_rmsIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_rmsprop_d2_bias_rmsIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_rmsprop_d3_kernel_rmsIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_rmsprop_d3_bias_rmsIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp-assignvariableop_19_rmsprop_output_kernel_rmsIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_rmsprop_output_bias_rmsIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_rmsprop_d1_kernel_momentumIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_rmsprop_d1_bias_momentumIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp.assignvariableop_23_rmsprop_d2_kernel_momentumIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_rmsprop_d2_bias_momentumIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_rmsprop_d3_kernel_momentumIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_rmsprop_d3_bias_momentumIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp2assignvariableop_27_rmsprop_output_kernel_momentumIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp0assignvariableop_28_rmsprop_output_bias_momentumIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_rmsprop_d1_kernel_mgIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_rmsprop_d1_bias_mgIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_rmsprop_d2_kernel_mgIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_rmsprop_d2_bias_mgIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_rmsprop_d3_kernel_mgIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_rmsprop_d3_bias_mgIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp,assignvariableop_35_rmsprop_output_kernel_mgIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp*assignvariableop_36_rmsprop_output_bias_mgIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37�
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�1
�
@__inference_model_layer_call_and_return_conditional_losses_73719
inputs_0
inputs_1%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource%
!d2_matmul_readvariableop_resource&
"d2_biasadd_readvariableop_resource%
!d3_matmul_readvariableop_resource&
"d3_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��d1/BiasAdd/ReadVariableOp�d1/MatMul/ReadVariableOp�d2/BiasAdd/ReadVariableOp�d2/MatMul/ReadVariableOp�d3/BiasAdd/ReadVariableOp�d3/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_1/Const�
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  2
flatten/Const�
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshapej
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat/axis�
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concat/concat�
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
d1/MatMul/ReadVariableOp�
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
	d1/MatMul�
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
d1/BiasAdd/ReadVariableOp�

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2	
d1/Relu�
dropout_d1/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_d1/Identityh
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip1/concat/axis�
skip1/concatConcatV2concat/concat:output:0dropout_d1/Identity:output:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	2
skip1/concat�
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
d2/MatMul/ReadVariableOp�
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
	d2/MatMul�
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
d2/BiasAdd/ReadVariableOp�

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2	
d2/Relu�
dropout_d2/IdentityIdentityd2/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_d2/Identity�
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
d3/MatMul/ReadVariableOp�
	d3/MatMulMatMuldropout_d2/Identity:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
	d3/MatMul�
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
d3/BiasAdd/ReadVariableOp�

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

d3/BiasAddb
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2	
d3/Relu�
dropout_d3/IdentityIdentityd3/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_d3/Identity�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldropout_d3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAddv
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output/Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentityoutput/Sigmoid:y:0^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp26
d3/BiasAdd/ReadVariableOpd3/BiasAdd/ReadVariableOp24
d3/MatMul/ReadVariableOpd3/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_73387

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
c
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73724

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�,
�
@__inference_model_layer_call_and_return_conditional_losses_74188
	mhc_input
peptide_input%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2%
!d3_statefulpartitionedcall_args_1%
!d3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_740052
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_733872
flatten/PartitionedCall�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_734372
concat/PartitionedCall�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_739152
d1/StatefulPartitionedCall�
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_737242
dropout_d1/PartitionedCall�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������	*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_736482
skip1/PartitionedCall�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_739342
d2/StatefulPartitionedCall�
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_737932
dropout_d2/PartitionedCall�
d3/StatefulPartitionedCallStatefulPartitionedCall#dropout_d2/PartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_737812
d3/StatefulPartitionedCall�
dropout_d3/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_736362
dropout_d3/PartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d3/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_740212 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
�+
�
!__inference__wrapped_model_486434
	mhc_input
peptide_input+
'model_d1_statefulpartitionedcall_args_1+
'model_d1_statefulpartitionedcall_args_2+
'model_d2_statefulpartitionedcall_args_1+
'model_d2_statefulpartitionedcall_args_2+
'model_d3_statefulpartitionedcall_args_1+
'model_d3_statefulpartitionedcall_args_2/
+model_output_statefulpartitionedcall_args_1/
+model_output_statefulpartitionedcall_args_2
identity�� model/d1/StatefulPartitionedCall� model/d2/StatefulPartitionedCall� model/d3/StatefulPartitionedCall�$model/output/StatefulPartitionedCall�
model/flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_743772!
model/flatten_1/PartitionedCall�
model/flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_743852
model/flatten/PartitionedCall�
model/concat/PartitionedCallPartitionedCall(model/flatten_1/PartitionedCall:output:0&model/flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_743942
model/concat/PartitionedCall�
 model/d1/StatefulPartitionedCallStatefulPartitionedCall%model/concat/PartitionedCall:output:0'model_d1_statefulpartitionedcall_args_1'model_d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_744042"
 model/d1/StatefulPartitionedCall�
 model/dropout_d1/PartitionedCallPartitionedCall)model/d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_744262"
 model/dropout_d1/PartitionedCall�
model/skip1/PartitionedCallPartitionedCall%model/concat/PartitionedCall:output:0)model/dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������	*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_744392
model/skip1/PartitionedCall�
 model/d2/StatefulPartitionedCallStatefulPartitionedCall$model/skip1/PartitionedCall:output:0'model_d2_statefulpartitionedcall_args_1'model_d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_744492"
 model/d2/StatefulPartitionedCall�
 model/dropout_d2/PartitionedCallPartitionedCall)model/d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_744712"
 model/dropout_d2/PartitionedCall�
 model/d3/StatefulPartitionedCallStatefulPartitionedCall)model/dropout_d2/PartitionedCall:output:0'model_d3_statefulpartitionedcall_args_1'model_d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_744852"
 model/d3/StatefulPartitionedCall�
 model/dropout_d3/PartitionedCallPartitionedCall)model/d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_745072"
 model/dropout_d3/PartitionedCall�
$model/output/StatefulPartitionedCallStatefulPartitionedCall)model/dropout_d3/PartitionedCall:output:0+model_output_statefulpartitionedcall_args_1+model_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_restored_function_body_745212&
$model/output/StatefulPartitionedCall�
IdentityIdentity-model/output/StatefulPartitionedCall:output:0!^model/d1/StatefulPartitionedCall!^model/d2/StatefulPartitionedCall!^model/d3/StatefulPartitionedCall%^model/output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::2D
 model/d1/StatefulPartitionedCall model/d1/StatefulPartitionedCall2D
 model/d2/StatefulPartitionedCall model/d2/StatefulPartitionedCall2D
 model/d3/StatefulPartitionedCall model/d3/StatefulPartitionedCall2L
$model/output/StatefulPartitionedCall$model/output/StatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
�
d
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73463

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
"__inference_d3_layer_call_fn_73788

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_737812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
&__inference_output_layer_call_fn_74136

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_740212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�1
�
@__inference_model_layer_call_and_return_conditional_losses_74162
	mhc_input
peptide_input%
!d1_statefulpartitionedcall_args_1%
!d1_statefulpartitionedcall_args_2%
!d2_statefulpartitionedcall_args_1%
!d2_statefulpartitionedcall_args_2%
!d3_statefulpartitionedcall_args_1%
!d3_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��d1/StatefulPartitionedCall�d2/StatefulPartitionedCall�d3/StatefulPartitionedCall�"dropout_d1/StatefulPartitionedCall�"dropout_d2/StatefulPartitionedCall�"dropout_d3/StatefulPartitionedCall�output/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_740052
flatten_1/PartitionedCall�
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_733872
flatten/PartitionedCall�
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_734372
concat/PartitionedCall�
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d1_layer_call_and_return_conditional_losses_739152
d1/StatefulPartitionedCall�
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d1_layer_call_and_return_conditional_losses_738182$
"dropout_d1/StatefulPartitionedCall�
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������	*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_736482
skip1/PartitionedCall�
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d2_layer_call_and_return_conditional_losses_739342
d2/StatefulPartitionedCall�
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_738862$
"dropout_d2/StatefulPartitionedCall�
d3/StatefulPartitionedCallStatefulPartitionedCall+dropout_d2/StatefulPartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*F
fAR?
=__inference_d3_layer_call_and_return_conditional_losses_737812
d3/StatefulPartitionedCall�
"dropout_d3/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0#^dropout_d2/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d3_layer_call_and_return_conditional_losses_736112$
"dropout_d3/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d3/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_740212 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^d2/StatefulPartitionedCall^d3/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall#^dropout_d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2H
"dropout_d3/StatefulPartitionedCall"dropout_d3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
�	
�
=__inference_d3_layer_call_and_return_conditional_losses_73781

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
R
(__inference_restored_function_body_74394

inputs
inputs_1
identity�
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_737362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
(__inference_restored_function_body_74521

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_output_layer_call_and_return_conditional_losses_736312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
R
(__inference_restored_function_body_74439

inputs
inputs_1
identity�
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*R
_output_shapes@
>:����������	: :����������:����������*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_skip1_layer_call_and_return_conditional_losses_734192
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
d
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73412

inputs
identity�a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/random_uniform/max�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniform�
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub�
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:����������2
dropout/random_uniform/mul�
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:����������2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv�
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:����������2
dropout/mul�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�
�
>__forward_skip1_layer_call_and_return_conditional_losses_75242

inputs_0_0

inputs_1_0
identity
concat_axis
inputs_0
inputs_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2
inputs_0_0
inputs_1_0concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������	2

Identity"#
concat_axisconcat/axis:output:0"
identityIdentity:output:0"
inputs_0
inputs_0_0"
inputs_1
inputs_1_0*;
_input_shapes*
(:����������:����������*m
backward_function_nameSQ__inference___backward_skip1_layer_call_and_return_conditional_losses_75224_75243:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_73430

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������":& "
 
_user_specified_nameinputs
�
l
@__inference_skip1_layer_call_and_return_conditional_losses_73419
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
;__forward_d3_layer_call_and_return_conditional_losses_75106
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
relu
matmul_readvariableop

inputs��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
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
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
reluRelu:activations:0*/
_input_shapes
:����������::*j
backward_function_namePN__inference___backward_d3_layer_call_and_return_conditional_losses_75092_7510720
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_486459
	mhc_input
peptide_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_4864342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:���������":���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
�
j
@__inference_skip1_layer_call_and_return_conditional_losses_73648

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������	2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������:����������:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�

�
=__inference_d2_layer_call_and_return_conditional_losses_73934

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu
d2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/Const�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
F
*__inference_dropout_d2_layer_call_fn_73798

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_d2_layer_call_and_return_conditional_losses_737932
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:& "
 
_user_specified_nameinputs
�

�
=__inference_d1_layer_call_and_return_conditional_losses_73571

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs"�L
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:߱
�K
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
layer-11
layer_with_weights-3
layer-12
	optimizer

signatures
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�H
_tf_keras_model�G{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}, "name": "mhc_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}, "name": "peptide_input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mhc_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["peptide_input", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concat", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip1", "inbound_nodes": [[["concat", 0, 0, {}], ["dropout_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d2", "inbound_nodes": [[["skip1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d2", "inbound_nodes": [[["d2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d3", "inbound_nodes": [[["dropout_d2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d3", "inbound_nodes": [[["d3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d3", 0, 0, {}]]]}], "input_layers": [["mhc_input", 0, 0], ["peptide_input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}, "name": "mhc_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}, "name": "peptide_input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mhc_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["peptide_input", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concat", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip1", "inbound_nodes": [[["concat", 0, 0, {}], ["dropout_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d2", "inbound_nodes": [[["skip1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d2", "inbound_nodes": [[["d2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d3", "inbound_nodes": [[["dropout_d2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d3", "inbound_nodes": [[["d3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d3", 0, 0, {}]]]}], "input_layers": [["mhc_input", 0, 0], ["peptide_input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "loss", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001038291840814054, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.5, "epsilon": 5.518842219864054e-07, "centered": true}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "mhc_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 34, 20], "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "peptide_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 15, 20], "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}}
�
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}}
�

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 980}}}}
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_d1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "skip1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}}
�

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "d2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1236}}}}
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_d2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "d3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_d3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
�
Hiter
	Idecay
Jlearning_rate
Kmomentum
Lrho	 rms}	!rms~	.rms
/rms�
8rms�
9rms�
Brms�
Crms� momentum�!momentum�.momentum�/momentum�8momentum�9momentum�Bmomentum�Cmomentum�	 mg�	!mg�	.mg�	/mg�	8mg�	9mg�	Bmg�	Cmg�"
	optimizer
-
�serving_default"
signature_map
X
 0
!1
.2
/3
84
95
B6
C7"
trackable_list_wrapper
X
 0
!1
.2
/3
84
95
B6
C7"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
Mlayer_regularization_losses
	variables
Nmetrics

Olayers
trainable_variables
Pnon_trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qlayer_regularization_losses
	variables
Rmetrics

Slayers
trainable_variables
Tnon_trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ulayer_regularization_losses
	variables
Vmetrics

Wlayers
trainable_variables
Xnon_trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ylayer_regularization_losses
	variables
Zmetrics

[layers
trainable_variables
\non_trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:
��2	d1/kernel
:�2d1/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
]layer_regularization_losses
"	variables
^metrics

_layers
#trainable_variables
`non_trainable_variables
$regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
alayer_regularization_losses
&	variables
bmetrics

clayers
'trainable_variables
dnon_trainable_variables
(regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
elayer_regularization_losses
*	variables
fmetrics

glayers
+trainable_variables
hnon_trainable_variables
,regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:
�	�2	d2/kernel
:�2d2/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
ilayer_regularization_losses
0	variables
jmetrics

klayers
1trainable_variables
lnon_trainable_variables
2regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mlayer_regularization_losses
4	variables
nmetrics

olayers
5trainable_variables
pnon_trainable_variables
6regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:
��2	d3/kernel
:�2d3/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
qlayer_regularization_losses
:	variables
rmetrics

slayers
;trainable_variables
tnon_trainable_variables
<regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ulayer_regularization_losses
>	variables
vmetrics

wlayers
?trainable_variables
xnon_trainable_variables
@regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2output/kernel
:2output/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ylayer_regularization_losses
D	variables
zmetrics

{layers
Etrainable_variables
|non_trainable_variables
Fregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
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
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
':%
��2RMSprop/d1/kernel/rms
 :�2RMSprop/d1/bias/rms
':%
�	�2RMSprop/d2/kernel/rms
 :�2RMSprop/d2/bias/rms
':%
��2RMSprop/d3/kernel/rms
 :�2RMSprop/d3/bias/rms
*:(	�2RMSprop/output/kernel/rms
#:!2RMSprop/output/bias/rms
,:*
��2RMSprop/d1/kernel/momentum
%:#�2RMSprop/d1/bias/momentum
,:*
�	�2RMSprop/d2/kernel/momentum
%:#�2RMSprop/d2/bias/momentum
,:*
��2RMSprop/d3/kernel/momentum
%:#�2RMSprop/d3/bias/momentum
/:-	�2RMSprop/output/kernel/momentum
(:&2RMSprop/output/bias/momentum
&:$
��2RMSprop/d1/kernel/mg
:�2RMSprop/d1/bias/mg
&:$
�	�2RMSprop/d2/kernel/mg
:�2RMSprop/d2/bias/mg
&:$
��2RMSprop/d3/kernel/mg
:�2RMSprop/d3/bias/mg
):'	�2RMSprop/output/kernel/mg
": 2RMSprop/output/bias/mg
�2�
@__inference_model_layer_call_and_return_conditional_losses_73719
@__inference_model_layer_call_and_return_conditional_losses_74162
@__inference_model_layer_call_and_return_conditional_losses_73559
@__inference_model_layer_call_and_return_conditional_losses_74188�
���
FullArgSpec)
args!�
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
annotations� *
 
�2�
!__inference__wrapped_model_486434�
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
annotations� *^�[
Y�V
'�$
	mhc_input���������"
+�(
peptide_input���������
�2�
%__inference_model_layer_call_fn_74075
%__inference_model_layer_call_fn_74061
%__inference_model_layer_call_fn_74129
%__inference_model_layer_call_fn_74115�
���
FullArgSpec)
args!�
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
annotations� *
 
�2�
D__inference_flatten_1_layer_call_and_return_conditional_losses_73430�
���
FullArgSpec
args�

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
annotations� *
 
�2�
)__inference_flatten_1_layer_call_fn_74010�
���
FullArgSpec
args�

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
annotations� *
 
�2�
B__inference_flatten_layer_call_and_return_conditional_losses_73660�
���
FullArgSpec
args�

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
annotations� *
 
�2�
'__inference_flatten_layer_call_fn_73392�
���
FullArgSpec
args�

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
annotations� *
 
�2�
A__inference_concat_layer_call_and_return_conditional_losses_73736�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_concat_layer_call_fn_73443�
���
FullArgSpec
args�

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
annotations� *
 
�2�
=__inference_d1_layer_call_and_return_conditional_losses_73571�
���
FullArgSpec
args�

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
annotations� *
 
�2�
"__inference_d1_layer_call_fn_73922�
���
FullArgSpec
args�

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
annotations� *
 
�2�
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73468
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73412�
���
FullArgSpec!
args�
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
annotations� *
 
�2�
*__inference_dropout_d1_layer_call_fn_73729
*__inference_dropout_d1_layer_call_fn_73823�
���
FullArgSpec!
args�
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
annotations� *
 
�2�
@__inference_skip1_layer_call_and_return_conditional_losses_73419�
���
FullArgSpec
args�

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
annotations� *
 
�2�
%__inference_skip1_layer_call_fn_73654�
���
FullArgSpec
args�

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
annotations� *
 
�2�
=__inference_d2_layer_call_and_return_conditional_losses_73903�
���
FullArgSpec
args�

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
annotations� *
 
�2�
"__inference_d2_layer_call_fn_73941�
���
FullArgSpec
args�

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
annotations� *
 
�2�
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73591
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73669�
���
FullArgSpec!
args�
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
annotations� *
 
�2�
*__inference_dropout_d2_layer_call_fn_73798
*__inference_dropout_d2_layer_call_fn_73891�
���
FullArgSpec!
args�
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
annotations� *
 
�2�
=__inference_d3_layer_call_and_return_conditional_losses_73834�
���
FullArgSpec
args�

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
annotations� *
 
�2�
"__inference_d3_layer_call_fn_73788�
���
FullArgSpec
args�

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
annotations� *
 
�2�
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73463
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73424�
���
FullArgSpec!
args�
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
annotations� *
 
�2�
*__inference_dropout_d3_layer_call_fn_73616
*__inference_dropout_d3_layer_call_fn_73641�
���
FullArgSpec!
args�
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
annotations� *
 
�2�
A__inference_output_layer_call_and_return_conditional_losses_73631�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_output_layer_call_fn_74136�
���
FullArgSpec
args�

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
annotations� *
 
BB@
$__inference_signature_wrapper_486459	mhc_inputpeptide_input
�2�
__inference_loss_fn_0_73664�
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
__inference_loss_fn_1_73673�
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
annotations� *� �
!__inference__wrapped_model_486434� !./89BCh�e
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
A__inference_concat_layer_call_and_return_conditional_losses_73736�\�Y
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
&__inference_concat_layer_call_fn_73443y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
=__inference_d1_layer_call_and_return_conditional_losses_73571^ !0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� w
"__inference_d1_layer_call_fn_73922Q !0�-
&�#
!�
inputs����������
� "������������
=__inference_d2_layer_call_and_return_conditional_losses_73903^./0�-
&�#
!�
inputs����������	
� "&�#
�
0����������
� w
"__inference_d2_layer_call_fn_73941Q./0�-
&�#
!�
inputs����������	
� "������������
=__inference_d3_layer_call_and_return_conditional_losses_73834^890�-
&�#
!�
inputs����������
� "&�#
�
0����������
� w
"__inference_d3_layer_call_fn_73788Q890�-
&�#
!�
inputs����������
� "������������
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73412^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
E__inference_dropout_d1_layer_call_and_return_conditional_losses_73468^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� 
*__inference_dropout_d1_layer_call_fn_73729Q4�1
*�'
!�
inputs����������
p 
� "�����������
*__inference_dropout_d1_layer_call_fn_73823Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73591^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
E__inference_dropout_d2_layer_call_and_return_conditional_losses_73669^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� 
*__inference_dropout_d2_layer_call_fn_73798Q4�1
*�'
!�
inputs����������
p 
� "�����������
*__inference_dropout_d2_layer_call_fn_73891Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73424^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
E__inference_dropout_d3_layer_call_and_return_conditional_losses_73463^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� 
*__inference_dropout_d3_layer_call_fn_73616Q4�1
*�'
!�
inputs����������
p
� "�����������
*__inference_dropout_d3_layer_call_fn_73641Q4�1
*�'
!�
inputs����������
p 
� "������������
D__inference_flatten_1_layer_call_and_return_conditional_losses_73430]3�0
)�&
$�!
inputs���������"
� "&�#
�
0����������
� }
)__inference_flatten_1_layer_call_fn_74010P3�0
)�&
$�!
inputs���������"
� "������������
B__inference_flatten_layer_call_and_return_conditional_losses_73660]3�0
)�&
$�!
inputs���������
� "&�#
�
0����������
� {
'__inference_flatten_layer_call_fn_73392P3�0
)�&
$�!
inputs���������
� "�����������7
__inference_loss_fn_0_73664�

� 
� "� 7
__inference_loss_fn_1_73673�

� 
� "� �
@__inference_model_layer_call_and_return_conditional_losses_73559� !./89BCj�g
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
@__inference_model_layer_call_and_return_conditional_losses_73719� !./89BCj�g
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
@__inference_model_layer_call_and_return_conditional_losses_74162� !./89BCp�m
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
@__inference_model_layer_call_and_return_conditional_losses_74188� !./89BCp�m
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
%__inference_model_layer_call_fn_74061� !./89BCp�m
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
%__inference_model_layer_call_fn_74075� !./89BCj�g
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
%__inference_model_layer_call_fn_74115� !./89BCp�m
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
%__inference_model_layer_call_fn_74129� !./89BCj�g
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
A__inference_output_layer_call_and_return_conditional_losses_73631]BC0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� z
&__inference_output_layer_call_fn_74136PBC0�-
&�#
!�
inputs����������
� "�����������
$__inference_signature_wrapper_486459� !./89BC��~
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
@__inference_skip1_layer_call_and_return_conditional_losses_73419�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������	
� �
%__inference_skip1_layer_call_fn_73654y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "�����������	