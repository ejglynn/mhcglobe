??
??
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
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ē
n
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name	d1/kernel
g
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel*
_output_shapes

:@*
dtype0
f
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	d1/bias
_
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
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
?
RMSprop/d1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameRMSprop/d1/kernel/rms

)RMSprop/d1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/rms*
_output_shapes

:@*
dtype0
~
RMSprop/d1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameRMSprop/d1/bias/rms
w
'RMSprop/d1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameRMSprop/output/kernel/rms
?
-RMSprop/output/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/rms*
_output_shapes

:*
dtype0
?
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
?
RMSprop/d1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_nameRMSprop/d1/kernel/momentum
?
.RMSprop/d1/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/momentum*
_output_shapes

:@*
dtype0
?
RMSprop/d1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/d1/bias/momentum
?
,RMSprop/d1/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/momentum*
_output_shapes
:*
dtype0
?
RMSprop/output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name RMSprop/output/kernel/momentum
?
2RMSprop/output/kernel/momentum/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/momentum*
_output_shapes

:*
dtype0
?
RMSprop/output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/output/bias/momentum
?
0RMSprop/output/bias/momentum/Read/ReadVariableOpReadVariableOpRMSprop/output/bias/momentum*
_output_shapes
:*
dtype0
?
RMSprop/d1/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameRMSprop/d1/kernel/mg
}
(RMSprop/d1/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/kernel/mg*
_output_shapes

:@*
dtype0
|
RMSprop/d1/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameRMSprop/d1/bias/mg
u
&RMSprop/d1/bias/mg/Read/ReadVariableOpReadVariableOpRMSprop/d1/bias/mg*
_output_shapes
:*
dtype0
?
RMSprop/output/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameRMSprop/output/kernel/mg
?
,RMSprop/output/kernel/mg/Read/ReadVariableOpReadVariableOpRMSprop/output/kernel/mg*
_output_shapes

:*
dtype0
?
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
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value?!B?! B?!
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?
iter
	decay
learning_rate
momentum
rho	rms9	rms:	rms;	rms<momentum=momentum>momentum?momentum@mgAmgBmgCmgD

0
1
2
3

0
1
2
3
 
?
 metrics
!layer_metrics
trainable_variables
"non_trainable_variables

#layers
$layer_regularization_losses
	variables
regularization_losses
 
US
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
%metrics
&layer_metrics
'non_trainable_variables
trainable_variables

(layers
)layer_regularization_losses
regularization_losses
	variables
 
 
 
?
*metrics
+layer_metrics
,non_trainable_variables
trainable_variables

-layers
.layer_regularization_losses
regularization_losses
	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
/metrics
0layer_metrics
1non_trainable_variables
trainable_variables

2layers
3layer_regularization_losses
regularization_losses
	variables
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

40
 
 

0
1
2
3
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
4
	5total
	6count
7	variables
8	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

50
61

7	variables
}
VARIABLE_VALUERMSprop/d1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUERMSprop/d1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/output/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/output/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/d1/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/d1/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/output/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/output/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/d1/kernel/mgSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUERMSprop/d1/bias/mgQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUERMSprop/output/kernel/mgSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUERMSprop/output/bias/mgQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE

serving_default_level2_inputPlaceholder*'
_output_shapes
:?????????@*
dtype0*
shape:?????????@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_level2_input	d1/kerneld1/biasoutput/kerneloutput/bias*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference_signature_wrapper_281635
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)RMSprop/d1/kernel/rms/Read/ReadVariableOp'RMSprop/d1/bias/rms/Read/ReadVariableOp-RMSprop/output/kernel/rms/Read/ReadVariableOp+RMSprop/output/bias/rms/Read/ReadVariableOp.RMSprop/d1/kernel/momentum/Read/ReadVariableOp,RMSprop/d1/bias/momentum/Read/ReadVariableOp2RMSprop/output/kernel/momentum/Read/ReadVariableOp0RMSprop/output/bias/momentum/Read/ReadVariableOp(RMSprop/d1/kernel/mg/Read/ReadVariableOp&RMSprop/d1/bias/mg/Read/ReadVariableOp,RMSprop/output/kernel/mg/Read/ReadVariableOp*RMSprop/output/bias/mg/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__traced_save_281878
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/biasoutput/kerneloutput/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/d1/kernel/rmsRMSprop/d1/bias/rmsRMSprop/output/kernel/rmsRMSprop/output/bias/rmsRMSprop/d1/kernel/momentumRMSprop/d1/bias/momentumRMSprop/output/kernel/momentumRMSprop/output/bias/momentumRMSprop/d1/kernel/mgRMSprop/d1/bias/mgRMSprop/output/kernel/mgRMSprop/output/bias/mg*#
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__traced_restore_281959??
?	
?
>__inference_d1_layer_call_and_return_conditional_losses_281461

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_281635
level2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllevel2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__wrapped_model_2814452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????@
&
_user_specified_namelevel2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281494

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?h
?
"__inference__traced_restore_281959
file_prefix
assignvariableop_d1_kernel
assignvariableop_1_d1_bias$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias#
assignvariableop_4_rmsprop_iter$
 assignvariableop_5_rmsprop_decay,
(assignvariableop_6_rmsprop_learning_rate'
#assignvariableop_7_rmsprop_momentum"
assignvariableop_8_rmsprop_rho
assignvariableop_9_total
assignvariableop_10_count-
)assignvariableop_11_rmsprop_d1_kernel_rms+
'assignvariableop_12_rmsprop_d1_bias_rms1
-assignvariableop_13_rmsprop_output_kernel_rms/
+assignvariableop_14_rmsprop_output_bias_rms2
.assignvariableop_15_rmsprop_d1_kernel_momentum0
,assignvariableop_16_rmsprop_d1_bias_momentum6
2assignvariableop_17_rmsprop_output_kernel_momentum4
0assignvariableop_18_rmsprop_output_bias_momentum,
(assignvariableop_19_rmsprop_d1_kernel_mg*
&assignvariableop_20_rmsprop_d1_bias_mg0
,assignvariableop_21_rmsprop_output_kernel_mg.
*assignvariableop_22_rmsprop_output_bias_mg
identity_24??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_d1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_d1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_rmsprop_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_rmsprop_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_rmsprop_momentumIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_rhoIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_rmsprop_d1_kernel_rmsIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp'assignvariableop_12_rmsprop_d1_bias_rmsIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_rmsprop_output_kernel_rmsIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_rmsprop_output_bias_rmsIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_rmsprop_d1_kernel_momentumIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_rmsprop_d1_bias_momentumIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_rmsprop_output_kernel_momentumIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_rmsprop_output_bias_momentumIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_rmsprop_d1_kernel_mgIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_rmsprop_d1_bias_mgIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp,assignvariableop_21_rmsprop_output_kernel_mgIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_rmsprop_output_bias_mgIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
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
NoOp?
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23?
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281742

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_d1_layer_call_fn_281757

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_2814942
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_281695

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2815712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_281552
level2_input
	d1_281539
	d1_281541
output_281545
output_281547
identity??d1/StatefulPartitionedCall?output/StatefulPartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCalllevel2_input	d1_281539	d1_281541*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_2814612
d1/StatefulPartitionedCall?
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_2814942
dropout_d1/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d1/PartitionedCall:output:0output_281545output_281547*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2815182 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:U Q
'
_output_shapes
:?????????@
&
_user_specified_namelevel2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_281571

inputs
	d1_281558
	d1_281560
output_281564
output_281566
identity??d1/StatefulPartitionedCall?"dropout_d1/StatefulPartitionedCall?output/StatefulPartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallinputs	d1_281558	d1_281560*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_2814612
d1/StatefulPartitionedCall?
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_2814892$
"dropout_d1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d1/StatefulPartitionedCall:output:0output_281564output_281566*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2815182 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_model_layer_call_fn_281611
level2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllevel2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2816002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????@
&
_user_specified_namelevel2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_output_layer_call_and_return_conditional_losses_281518

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281489

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
x
#__inference_d1_layer_call_fn_281730

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_2814612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_output_layer_call_fn_281777

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2815182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_281682

inputs%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
d1/MatMul/ReadVariableOp|
	d1/MatMulMatMulinputs d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

d1/BiasAddj

d1/SigmoidSigmoidd1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

d1/Sigmoidx
dropout_d1/IdentityIdentityd1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
dropout_d1/Identity?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_d1/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Constf
IdentityIdentityoutput/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_281662

inputs%
!d1_matmul_readvariableop_resource&
"d1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
d1/MatMul/ReadVariableOp|
	d1/MatMulMatMulinputs d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

d1/BiasAddj

d1/SigmoidSigmoidd1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

d1/Sigmoidy
dropout_d1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_d1/dropout/Const?
dropout_d1/dropout/MulMuld1/Sigmoid:y:0!dropout_d1/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_d1/dropout/Mulr
dropout_d1/dropout/ShapeShaped1/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_d1/dropout/Shape?
/dropout_d1/dropout/random_uniform/RandomUniformRandomUniform!dropout_d1/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype021
/dropout_d1/dropout/random_uniform/RandomUniform?
!dropout_d1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_d1/dropout/GreaterEqual/y?
dropout_d1/dropout/GreaterEqualGreaterEqual8dropout_d1/dropout/random_uniform/RandomUniform:output:0*dropout_d1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2!
dropout_d1/dropout/GreaterEqual?
dropout_d1/dropout/CastCast#dropout_d1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_d1/dropout/Cast?
dropout_d1/dropout/Mul_1Muldropout_d1/dropout/Mul:z:0dropout_d1/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_d1/dropout/Mul_1?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_d1/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddv
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output/Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Constf
IdentityIdentityoutput/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
>__inference_d1_layer_call_and_return_conditional_losses_281721

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
d
+__inference_dropout_d1_layer_call_fn_281752

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_2814892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_model_layer_call_and_return_conditional_losses_281536
level2_input
	d1_281472
	d1_281474
output_281529
output_281531
identity??d1/StatefulPartitionedCall?"dropout_d1/StatefulPartitionedCall?output/StatefulPartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCalllevel2_input	d1_281472	d1_281474*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_2814612
d1/StatefulPartitionedCall?
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_2814892$
"dropout_d1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d1/StatefulPartitionedCall:output:0output_281529output_281531*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2815182 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:U Q
'
_output_shapes
:?????????@
&
_user_specified_namelevel2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_output_layer_call_and_return_conditional_losses_281768

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
,
__inference_loss_fn_0_281782
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
??
?

__inference__traced_save_281878
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop,
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
.savev2_rmsprop_d1_bias_rms_read_readvariableop8
4savev2_rmsprop_output_kernel_rms_read_readvariableop6
2savev2_rmsprop_output_bias_rms_read_readvariableop9
5savev2_rmsprop_d1_kernel_momentum_read_readvariableop7
3savev2_rmsprop_d1_bias_momentum_read_readvariableop=
9savev2_rmsprop_output_kernel_momentum_read_readvariableop;
7savev2_rmsprop_output_bias_momentum_read_readvariableop3
/savev2_rmsprop_d1_kernel_mg_read_readvariableop1
-savev2_rmsprop_d1_bias_mg_read_readvariableop7
3savev2_rmsprop_output_kernel_mg_read_readvariableop5
1savev2_rmsprop_output_bias_mg_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e35f21a5f2044af6ac9849578f901322/part2	
Const_1?
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_rmsprop_d1_kernel_rms_read_readvariableop.savev2_rmsprop_d1_bias_rms_read_readvariableop4savev2_rmsprop_output_kernel_rms_read_readvariableop2savev2_rmsprop_output_bias_rms_read_readvariableop5savev2_rmsprop_d1_kernel_momentum_read_readvariableop3savev2_rmsprop_d1_bias_momentum_read_readvariableop9savev2_rmsprop_output_kernel_momentum_read_readvariableop7savev2_rmsprop_output_bias_momentum_read_readvariableop/savev2_rmsprop_d1_kernel_mg_read_readvariableop-savev2_rmsprop_d1_bias_mg_read_readvariableop3savev2_rmsprop_output_kernel_mg_read_readvariableop1savev2_rmsprop_output_bias_mg_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:::: : : : : : : :@::::@::::@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
d
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281747

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_281445
level2_input+
'model_d1_matmul_readvariableop_resource,
(model_d1_biasadd_readvariableop_resource/
+model_output_matmul_readvariableop_resource0
,model_output_biasadd_readvariableop_resource
identity??
model/d1/MatMul/ReadVariableOpReadVariableOp'model_d1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
model/d1/MatMul/ReadVariableOp?
model/d1/MatMulMatMullevel2_input&model/d1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/d1/MatMul?
model/d1/BiasAdd/ReadVariableOpReadVariableOp(model_d1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
model/d1/BiasAdd/ReadVariableOp?
model/d1/BiasAddBiasAddmodel/d1/MatMul:product:0'model/d1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/d1/BiasAdd|
model/d1/SigmoidSigmoidmodel/d1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/d1/Sigmoid?
model/dropout_d1/IdentityIdentitymodel/d1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2
model/dropout_d1/Identity?
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"model/output/MatMul/ReadVariableOp?
model/output/MatMulMatMul"model/dropout_d1/Identity:output:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/output/MatMul?
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model/output/BiasAdd/ReadVariableOp?
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/output/BiasAdd?
model/output/SigmoidSigmoidmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/output/Sigmoidl
IdentityIdentitymodel/output/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::::U Q
'
_output_shapes
:?????????@
&
_user_specified_namelevel2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_model_layer_call_fn_281582
level2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllevel2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2815712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????@
&
_user_specified_namelevel2_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_281600

inputs
	d1_281587
	d1_281589
output_281593
output_281595
identity??d1/StatefulPartitionedCall?output/StatefulPartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallinputs	d1_281587	d1_281589*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_2814612
d1/StatefulPartitionedCall?
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_2814942
dropout_d1/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d1/PartitionedCall:output:0output_281593output_281595*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2815182 
output/StatefulPartitionedCall
d1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/Const?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_model_layer_call_fn_281708

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2816002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
level2_input5
serving_default_level2_input:0?????????@:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ƀ
? 
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
*E&call_and_return_all_conditional_losses
F__call__
G_default_save_signature"?
_tf_keras_model?{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "level2_input"}, "name": "level2_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 15, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["level2_input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d1", 0, 0, {}]]]}], "input_layers": [["level2_input", 0, 0]], "output_layers": [["output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "level2_input"}, "name": "level2_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 15, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["level2_input", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d1", 0, 0, {}]]]}], "input_layers": [["level2_input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "mae", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.800000011920929, "epsilon": 1e-07, "centered": true}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "level2_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "level2_input"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 15, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_d1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 15}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15]}}
?
iter
	decay
learning_rate
momentum
rho	rms9	rms:	rms;	rms<momentum=momentum>momentum?momentum@mgAmgBmgCmgD"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
?
 metrics
!layer_metrics
trainable_variables
"non_trainable_variables

#layers
$layer_regularization_losses
	variables
regularization_losses
F__call__
G_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Oserving_default"
signature_map
:@2	d1/kernel
:2d1/bias
.
0
1"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
%metrics
&layer_metrics
'non_trainable_variables
trainable_variables

(layers
)layer_regularization_losses
regularization_losses
	variables
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
*metrics
+layer_metrics
,non_trainable_variables
trainable_variables

-layers
.layer_regularization_losses
regularization_losses
	variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
:2output/kernel
:2output/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
/metrics
0layer_metrics
1non_trainable_variables
trainable_variables

2layers
3layer_regularization_losses
regularization_losses
	variables
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
'
40"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
'
N0"
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
?
	5total
	6count
7	variables
8	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
50
61"
trackable_list_wrapper
-
7	variables"
_generic_user_object
%:#@2RMSprop/d1/kernel/rms
:2RMSprop/d1/bias/rms
):'2RMSprop/output/kernel/rms
#:!2RMSprop/output/bias/rms
*:(@2RMSprop/d1/kernel/momentum
$:"2RMSprop/d1/bias/momentum
.:,2RMSprop/output/kernel/momentum
(:&2RMSprop/output/bias/momentum
$:"@2RMSprop/d1/kernel/mg
:2RMSprop/d1/bias/mg
(:&2RMSprop/output/kernel/mg
": 2RMSprop/output/bias/mg
?2?
A__inference_model_layer_call_and_return_conditional_losses_281662
A__inference_model_layer_call_and_return_conditional_losses_281682
A__inference_model_layer_call_and_return_conditional_losses_281536
A__inference_model_layer_call_and_return_conditional_losses_281552?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_model_layer_call_fn_281708
&__inference_model_layer_call_fn_281611
&__inference_model_layer_call_fn_281582
&__inference_model_layer_call_fn_281695?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_281445?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
level2_input?????????@
?2?
>__inference_d1_layer_call_and_return_conditional_losses_281721?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_d1_layer_call_fn_281730?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281742
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281747?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_d1_layer_call_fn_281752
+__inference_dropout_d1_layer_call_fn_281757?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_output_layer_call_and_return_conditional_losses_281768?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_output_layer_call_fn_281777?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_281782?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
8B6
$__inference_signature_wrapper_281635level2_input?
!__inference__wrapped_model_281445n5?2
+?(
&?#
level2_input?????????@
? "/?,
*
output ?
output??????????
>__inference_d1_layer_call_and_return_conditional_losses_281721\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? v
#__inference_d1_layer_call_fn_281730O/?,
%?"
 ?
inputs?????????@
? "???????????
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281742\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
F__inference_dropout_d1_layer_call_and_return_conditional_losses_281747\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ~
+__inference_dropout_d1_layer_call_fn_281752O3?0
)?&
 ?
inputs?????????
p
? "??????????~
+__inference_dropout_d1_layer_call_fn_281757O3?0
)?&
 ?
inputs?????????
p 
? "??????????8
__inference_loss_fn_0_281782?

? 
? "? ?
A__inference_model_layer_call_and_return_conditional_losses_281536l=?:
3?0
&?#
level2_input?????????@
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_281552l=?:
3?0
&?#
level2_input?????????@
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_281662f7?4
-?*
 ?
inputs?????????@
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_281682f7?4
-?*
 ?
inputs?????????@
p 

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_281582_=?:
3?0
&?#
level2_input?????????@
p

 
? "???????????
&__inference_model_layer_call_fn_281611_=?:
3?0
&?#
level2_input?????????@
p 

 
? "???????????
&__inference_model_layer_call_fn_281695Y7?4
-?*
 ?
inputs?????????@
p

 
? "???????????
&__inference_model_layer_call_fn_281708Y7?4
-?*
 ?
inputs?????????@
p 

 
? "???????????
B__inference_output_layer_call_and_return_conditional_losses_281768\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_output_layer_call_fn_281777O/?,
%?"
 ?
inputs?????????
? "???????????
$__inference_signature_wrapper_281635~E?B
? 
;?8
6
level2_input&?#
level2_input?????????@"/?,
*
output ?
output?????????