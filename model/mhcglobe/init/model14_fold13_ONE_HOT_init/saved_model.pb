??
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
shapeshape?"serve*2.1.02v2.1.0-6-g2dd7e988??	
p
	d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	d1/kernel
i
d1/kernel/Read/ReadVariableOpReadVariableOp	d1/kernel* 
_output_shapes
:
??*
dtype0
g
d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	d1/bias
`
d1/bias/Read/ReadVariableOpReadVariableOpd1/bias*
_output_shapes	
:?*
dtype0
p
	d2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name	d2/kernel
i
d2/kernel/Read/ReadVariableOpReadVariableOp	d2/kernel* 
_output_shapes
:
??*
dtype0
g
d2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	d2/bias
`
d2/bias/Read/ReadVariableOpReadVariableOpd2/bias*
_output_shapes	
:?*
dtype0
p
	d3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*
shared_name	d3/kernel
i
d3/kernel/Read/ReadVariableOpReadVariableOp	d3/kernel* 
_output_shapes
:
?	?*
dtype0
g
d3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	d3/bias
`
d3/bias/Read/ReadVariableOpReadVariableOpd3/bias*
_output_shapes	
:?*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	?*
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

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
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
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
 
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
R
'regularization_losses
(trainable_variables
)	variables
*	keras_api
R
+regularization_losses
,trainable_variables
-	variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
R
5regularization_losses
6trainable_variables
7	variables
8	keras_api
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
R
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
 
 
8
!0
"1
/2
03
=4
>5
G6
H7
8
!0
"1
/2
03
=4
>5
G6
H7
?
regularization_losses
Mmetrics

Nlayers
Olayer_regularization_losses
Pnon_trainable_variables
trainable_variables
	variables
 
 
 
 
?
regularization_losses
Qmetrics

Rlayers
Slayer_regularization_losses
Tnon_trainable_variables
trainable_variables
	variables
 
 
 
?
regularization_losses
Umetrics

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
trainable_variables
	variables
 
 
 
?
regularization_losses
Ymetrics

Zlayers
[layer_regularization_losses
\non_trainable_variables
trainable_variables
	variables
US
VARIABLE_VALUE	d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
?
#regularization_losses
]metrics

^layers
_layer_regularization_losses
`non_trainable_variables
$trainable_variables
%	variables
 
 
 
?
'regularization_losses
ametrics

blayers
clayer_regularization_losses
dnon_trainable_variables
(trainable_variables
)	variables
 
 
 
?
+regularization_losses
emetrics

flayers
glayer_regularization_losses
hnon_trainable_variables
,trainable_variables
-	variables
US
VARIABLE_VALUE	d2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
1regularization_losses
imetrics

jlayers
klayer_regularization_losses
lnon_trainable_variables
2trainable_variables
3	variables
 
 
 
?
5regularization_losses
mmetrics

nlayers
olayer_regularization_losses
pnon_trainable_variables
6trainable_variables
7	variables
 
 
 
?
9regularization_losses
qmetrics

rlayers
slayer_regularization_losses
tnon_trainable_variables
:trainable_variables
;	variables
US
VARIABLE_VALUE	d3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEd3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?
?regularization_losses
umetrics

vlayers
wlayer_regularization_losses
xnon_trainable_variables
@trainable_variables
A	variables
 
 
 
?
Cregularization_losses
ymetrics

zlayers
{layer_regularization_losses
|non_trainable_variables
Dtrainable_variables
E	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
?
Iregularization_losses
}metrics

~layers
layer_regularization_losses
?non_trainable_variables
Jtrainable_variables
K	variables
 
f
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
 
 
 
 
 
?
serving_default_mhc_inputPlaceholder*+
_output_shapes
:?????????"*
dtype0* 
shape:?????????"
?
serving_default_peptide_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_mhc_inputserving_default_peptide_input	d1/kerneld1/bias	d2/kerneld2/bias	d3/kerneld3/biasoutput/kerneloutput/bias*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_191300
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamed1/kernel/Read/ReadVariableOpd1/bias/Read/ReadVariableOpd2/kernel/Read/ReadVariableOpd2/bias/Read/ReadVariableOpd3/kernel/Read/ReadVariableOpd3/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpConst*
Tin
2
*
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
__inference__traced_save_191852
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	d1/kerneld1/bias	d2/kerneld2/bias	d3/kerneld3/biasoutput/kerneloutput/bias*
Tin
2	*
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
"__inference__traced_restore_191888??
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_191514

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?'
?
"__inference__traced_restore_191888
file_prefix
assignvariableop_d1_kernel
assignvariableop_1_d1_bias 
assignvariableop_2_d2_kernel
assignvariableop_3_d2_bias 
assignvariableop_4_d3_kernel
assignvariableop_5_d3_bias$
 assignvariableop_6_output_kernel"
assignvariableop_7_output_bias

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
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
AssignVariableOp_2AssignVariableOpassignvariableop_2_d2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_d2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_d3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_d3_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_output_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_output_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7?
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
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
d
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191064

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout_d1_layer_call_and_return_conditional_losses_191586

inputs
identity?a
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
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout_d1_layer_call_and_return_conditional_losses_190897

inputs
identity?a
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
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
>__inference_d2_layer_call_and_return_conditional_losses_191641

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?(d2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp)^d2/kernel/Regularizer/Abs/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp:& "
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_190808

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????":& "
 
_user_specified_nameinputs
?
e
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191059

inputs
identity?a
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
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_1917805
1d1_kernel_regularizer_abs_readvariableop_resource
identity??(d1/kernel/Regularizer/Abs/ReadVariableOp?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp1d1_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
IdentityIdentityd1/kernel/Regularizer/add:z:0)^d1/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp
?

?
$__inference_signature_wrapper_191300
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_1907972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?
d
F__inference_dropout_d2_layer_call_and_return_conditional_losses_191673

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout_d2_layer_call_and_return_conditional_losses_191668

inputs
identity?a
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
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_191269
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1912582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?
d
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191739

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?D
?
A__inference_model_layer_call_and_return_conditional_losses_191158
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
identity??d1/StatefulPartitionedCall?(d1/kernel/Regularizer/Abs/ReadVariableOp?d2/StatefulPartitionedCall?(d2/kernel/Regularizer/Abs/ReadVariableOp?d3/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1908082
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1908222
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_concat_layer_call_and_return_conditional_losses_1908372
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_1908652
d1/StatefulPartitionedCall?
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_1909022
dropout_d1/PartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip1_layer_call_and_return_conditional_losses_1909222
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d2_layer_call_and_return_conditional_losses_1909502
d2/StatefulPartitionedCall?
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d2_layer_call_and_return_conditional_losses_1909872
dropout_d2/PartitionedCall?
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0#dropout_d2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip2_layer_call_and_return_conditional_losses_1910072
skip2/PartitionedCall?
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d3_layer_call_and_return_conditional_losses_1910272
d3/StatefulPartitionedCall?
dropout_d3/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d3_layer_call_and_return_conditional_losses_1910642
dropout_d3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d3/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1910882 
output/StatefulPartitionedCall?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_statefulpartitionedcall_args_1^d1/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d2_statefulpartitionedcall_args_1^d2/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall)^d1/kernel/Regularizer/Abs/ReadVariableOp^d2/StatefulPartitionedCall)^d2/kernel/Regularizer/Abs/ReadVariableOp^d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?
e
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191734

inputs
identity?a
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
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
!
cond_false_191821
identityz
ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_258de7774f2a4340928f25e2d7aa08c7/part2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
??
?
A__inference_model_layer_call_and_return_conditional_losses_191407
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
identity??d1/BiasAdd/ReadVariableOp?d1/MatMul/ReadVariableOp?(d1/kernel/Regularizer/Abs/ReadVariableOp?d2/BiasAdd/ReadVariableOp?d2/MatMul/ReadVariableOp?(d2/kernel/Regularizer/Abs/ReadVariableOp?d3/BiasAdd/ReadVariableOp?d3/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
flatten/Const?
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapej
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat/axis?
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat/concat?
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d1/MatMul/ReadVariableOp?
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
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
dropout_d1/dropout/Shape?
%dropout_d1/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d1/dropout/random_uniform/min?
%dropout_d1/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dropout_d1/dropout/random_uniform/max?
/dropout_d1/dropout/random_uniform/RandomUniformRandomUniform!dropout_d1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_d1/dropout/random_uniform/RandomUniform?
%dropout_d1/dropout/random_uniform/subSub.dropout_d1/dropout/random_uniform/max:output:0.dropout_d1/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d1/dropout/random_uniform/sub?
%dropout_d1/dropout/random_uniform/mulMul8dropout_d1/dropout/random_uniform/RandomUniform:output:0)dropout_d1/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2'
%dropout_d1/dropout/random_uniform/mul?
!dropout_d1/dropout/random_uniformAdd)dropout_d1/dropout/random_uniform/mul:z:0.dropout_d1/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2#
!dropout_d1/dropout/random_uniformy
dropout_d1/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d1/dropout/sub/x?
dropout_d1/dropout/subSub!dropout_d1/dropout/sub/x:output:0 dropout_d1/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d1/dropout/sub?
dropout_d1/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d1/dropout/truediv/x?
dropout_d1/dropout/truedivRealDiv%dropout_d1/dropout/truediv/x:output:0dropout_d1/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d1/dropout/truediv?
dropout_d1/dropout/GreaterEqualGreaterEqual%dropout_d1/dropout/random_uniform:z:0 dropout_d1/dropout/rate:output:0*
T0*(
_output_shapes
:??????????2!
dropout_d1/dropout/GreaterEqual?
dropout_d1/dropout/mulMuld1/Relu:activations:0dropout_d1/dropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout_d1/dropout/mul?
dropout_d1/dropout/CastCast#dropout_d1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_d1/dropout/Cast?
dropout_d1/dropout/mul_1Muldropout_d1/dropout/mul:z:0dropout_d1/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_d1/dropout/mul_1h
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip1/concat/axis?
skip1/concatConcatV2concat/concat:output:0dropout_d1/dropout/mul_1:z:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
skip1/concat?
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d2/MatMul/ReadVariableOp?
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d2/MatMul?
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d2/BiasAdd/ReadVariableOp?

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
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
dropout_d2/dropout/Shape?
%dropout_d2/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d2/dropout/random_uniform/min?
%dropout_d2/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dropout_d2/dropout/random_uniform/max?
/dropout_d2/dropout/random_uniform/RandomUniformRandomUniform!dropout_d2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_d2/dropout/random_uniform/RandomUniform?
%dropout_d2/dropout/random_uniform/subSub.dropout_d2/dropout/random_uniform/max:output:0.dropout_d2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d2/dropout/random_uniform/sub?
%dropout_d2/dropout/random_uniform/mulMul8dropout_d2/dropout/random_uniform/RandomUniform:output:0)dropout_d2/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2'
%dropout_d2/dropout/random_uniform/mul?
!dropout_d2/dropout/random_uniformAdd)dropout_d2/dropout/random_uniform/mul:z:0.dropout_d2/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2#
!dropout_d2/dropout/random_uniformy
dropout_d2/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d2/dropout/sub/x?
dropout_d2/dropout/subSub!dropout_d2/dropout/sub/x:output:0 dropout_d2/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d2/dropout/sub?
dropout_d2/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d2/dropout/truediv/x?
dropout_d2/dropout/truedivRealDiv%dropout_d2/dropout/truediv/x:output:0dropout_d2/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d2/dropout/truediv?
dropout_d2/dropout/GreaterEqualGreaterEqual%dropout_d2/dropout/random_uniform:z:0 dropout_d2/dropout/rate:output:0*
T0*(
_output_shapes
:??????????2!
dropout_d2/dropout/GreaterEqual?
dropout_d2/dropout/mulMuld2/Relu:activations:0dropout_d2/dropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout_d2/dropout/mul?
dropout_d2/dropout/CastCast#dropout_d2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_d2/dropout/Cast?
dropout_d2/dropout/mul_1Muldropout_d2/dropout/mul:z:0dropout_d2/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_d2/dropout/mul_1h
skip2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip2/concat/axis?
skip2/concatConcatV2skip1/concat:output:0dropout_d2/dropout/mul_1:z:0skip2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
skip2/concat?
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
d3/MatMul/ReadVariableOp?
	d3/MatMulMatMulskip2/concat:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d3/MatMul?
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d3/BiasAdd/ReadVariableOp?

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d3/BiasAddb
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
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
dropout_d3/dropout/Shape?
%dropout_d3/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_d3/dropout/random_uniform/min?
%dropout_d3/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dropout_d3/dropout/random_uniform/max?
/dropout_d3/dropout/random_uniform/RandomUniformRandomUniform!dropout_d3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_d3/dropout/random_uniform/RandomUniform?
%dropout_d3/dropout/random_uniform/subSub.dropout_d3/dropout/random_uniform/max:output:0.dropout_d3/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_d3/dropout/random_uniform/sub?
%dropout_d3/dropout/random_uniform/mulMul8dropout_d3/dropout/random_uniform/RandomUniform:output:0)dropout_d3/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2'
%dropout_d3/dropout/random_uniform/mul?
!dropout_d3/dropout/random_uniformAdd)dropout_d3/dropout/random_uniform/mul:z:0.dropout_d3/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2#
!dropout_d3/dropout/random_uniformy
dropout_d3/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d3/dropout/sub/x?
dropout_d3/dropout/subSub!dropout_d3/dropout/sub/x:output:0 dropout_d3/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_d3/dropout/sub?
dropout_d3/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_d3/dropout/truediv/x?
dropout_d3/dropout/truedivRealDiv%dropout_d3/dropout/truediv/x:output:0dropout_d3/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_d3/dropout/truediv?
dropout_d3/dropout/GreaterEqualGreaterEqual%dropout_d3/dropout/random_uniform:z:0 dropout_d3/dropout/rate:output:0*
T0*(
_output_shapes
:??????????2!
dropout_d3/dropout/GreaterEqual?
dropout_d3/dropout/mulMuld3/Relu:activations:0dropout_d3/dropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout_d3/dropout/mul?
dropout_d3/dropout/CastCast#dropout_d3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_d3/dropout/Cast?
dropout_d3/dropout/mul_1Muldropout_d3/dropout/mul:z:0dropout_d3/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_d3/dropout/mul_1?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_d3/dropout/mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
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
output/Sigmoid?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource^d1/MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource^d2/MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentityoutput/Sigmoid:y:0^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp)^d1/kernel/Regularizer/Abs/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp)^d2/kernel/Regularizer/Abs/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp26
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
?
m
A__inference_skip2_layer_call_and_return_conditional_losses_191690
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
R
&__inference_skip1_layer_call_fn_191614
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip1_layer_call_and_return_conditional_losses_1909222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
k
A__inference_skip2_layer_call_and_return_conditional_losses_191007

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?	
?
B__inference_output_layer_call_and_return_conditional_losses_191088

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout_d1_layer_call_and_return_conditional_losses_191591

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_1917935
1d2_kernel_regularizer_abs_readvariableop_resource
identity??(d2/kernel/Regularizer/Abs/ReadVariableOp?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp1d2_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentityd2/kernel/Regularizer/add:z:0)^d2/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp
?
k
A__inference_skip1_layer_call_and_return_conditional_losses_190922

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_191214
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	mhc_inputpeptide_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1912032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?
G
+__inference_dropout_d1_layer_call_fn_191601

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_1909022
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_191483
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1912032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
#__inference_d1_layer_call_fn_191566

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_1908652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout_d1_layer_call_and_return_conditional_losses_190902

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?I
?
A__inference_model_layer_call_and_return_conditional_losses_191203

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
identity??d1/StatefulPartitionedCall?(d1/kernel/Regularizer/Abs/ReadVariableOp?d2/StatefulPartitionedCall?(d2/kernel/Regularizer/Abs/ReadVariableOp?d3/StatefulPartitionedCall?"dropout_d1/StatefulPartitionedCall?"dropout_d2/StatefulPartitionedCall?"dropout_d3/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1908082
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1908222
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_concat_layer_call_and_return_conditional_losses_1908372
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_1908652
d1/StatefulPartitionedCall?
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_1908972$
"dropout_d1/StatefulPartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip1_layer_call_and_return_conditional_losses_1909222
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d2_layer_call_and_return_conditional_losses_1909502
d2/StatefulPartitionedCall?
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d2_layer_call_and_return_conditional_losses_1909822$
"dropout_d2/StatefulPartitionedCall?
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0+dropout_d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip2_layer_call_and_return_conditional_losses_1910072
skip2/PartitionedCall?
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d3_layer_call_and_return_conditional_losses_1910272
d3/StatefulPartitionedCall?
"dropout_d3/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0#^dropout_d2/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d3_layer_call_and_return_conditional_losses_1910592$
"dropout_d3/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d3/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1910882 
output/StatefulPartitionedCall?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_statefulpartitionedcall_args_1^d1/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d2_statefulpartitionedcall_args_1^d2/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall)^d1/kernel/Regularizer/Abs/ReadVariableOp^d2/StatefulPartitionedCall)^d2/kernel/Regularizer/Abs/ReadVariableOp^d3/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall#^dropout_d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2H
"dropout_d3/StatefulPartitionedCall"dropout_d3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_191497
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
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1912582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
G
+__inference_dropout_d3_layer_call_fn_191749

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d3_layer_call_and_return_conditional_losses_1910642
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
'__inference_output_layer_call_fn_191767

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1910882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
#__inference_d2_layer_call_fn_191648

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d2_layer_call_and_return_conditional_losses_1909502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_191503

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????":& "
 
_user_specified_nameinputs
?G
?
A__inference_model_layer_call_and_return_conditional_losses_191469
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
identity??d1/BiasAdd/ReadVariableOp?d1/MatMul/ReadVariableOp?(d1/kernel/Regularizer/Abs/ReadVariableOp?d2/BiasAdd/ReadVariableOp?d2/MatMul/ReadVariableOp?(d2/kernel/Regularizer/Abs/ReadVariableOp?d3/BiasAdd/ReadVariableOp?d3/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapeinputs_0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
flatten/Const?
flatten/ReshapeReshapeinputs_1flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapej
concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/concat/axis?
concat/concatConcatV2flatten_1/Reshape:output:0flatten/Reshape:output:0concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat/concat?
d1/MatMul/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d1/MatMul/ReadVariableOp?
	d1/MatMulMatMulconcat/concat:output:0 d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d1/MatMul?
d1/BiasAdd/ReadVariableOpReadVariableOp"d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d1/BiasAdd/ReadVariableOp?

d1/BiasAddBiasAddd1/MatMul:product:0!d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d1/BiasAddb
d1/ReluRelud1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d1/Relu?
dropout_d1/IdentityIdentityd1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_d1/Identityh
skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip1/concat/axis?
skip1/concatConcatV2concat/concat:output:0dropout_d1/Identity:output:0skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
skip1/concat?
d2/MatMul/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
d2/MatMul/ReadVariableOp?
	d2/MatMulMatMulskip1/concat:output:0 d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d2/MatMul?
d2/BiasAdd/ReadVariableOpReadVariableOp"d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d2/BiasAdd/ReadVariableOp?

d2/BiasAddBiasAddd2/MatMul:product:0!d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d2/BiasAddb
d2/ReluRelud2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d2/Relu?
dropout_d2/IdentityIdentityd2/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_d2/Identityh
skip2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
skip2/concat/axis?
skip2/concatConcatV2skip1/concat:output:0dropout_d2/Identity:output:0skip2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
skip2/concat?
d3/MatMul/ReadVariableOpReadVariableOp!d3_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
d3/MatMul/ReadVariableOp?
	d3/MatMulMatMulskip2/concat:output:0 d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	d3/MatMul?
d3/BiasAdd/ReadVariableOpReadVariableOp"d3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
d3/BiasAdd/ReadVariableOp?

d3/BiasAddBiasAddd3/MatMul:product:0!d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

d3/BiasAddb
d3/ReluRelud3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
d3/Relu?
dropout_d3/IdentityIdentityd3/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_d3/Identity?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldropout_d3/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
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
output/Sigmoid?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_matmul_readvariableop_resource^d1/MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d2_matmul_readvariableop_resource^d2/MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentityoutput/Sigmoid:y:0^d1/BiasAdd/ReadVariableOp^d1/MatMul/ReadVariableOp)^d1/kernel/Regularizer/Abs/ReadVariableOp^d2/BiasAdd/ReadVariableOp^d2/MatMul/ReadVariableOp)^d2/kernel/Regularizer/Abs/ReadVariableOp^d3/BiasAdd/ReadVariableOp^d3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::26
d1/BiasAdd/ReadVariableOpd1/BiasAdd/ReadVariableOp24
d1/MatMul/ReadVariableOpd1/MatMul/ReadVariableOp2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp26
d2/BiasAdd/ReadVariableOpd2/BiasAdd/ReadVariableOp24
d2/MatMul/ReadVariableOpd2/MatMul/ReadVariableOp2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp26
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
?
?
>__inference_d2_layer_call_and_return_conditional_losses_190950

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?(d2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp)^d2/kernel/Regularizer/Abs/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp:& "
 
_user_specified_nameinputs
?8
?
!__inference__wrapped_model_190797
	mhc_input
peptide_input+
'model_d1_matmul_readvariableop_resource,
(model_d1_biasadd_readvariableop_resource+
'model_d2_matmul_readvariableop_resource,
(model_d2_biasadd_readvariableop_resource+
'model_d3_matmul_readvariableop_resource,
(model_d3_biasadd_readvariableop_resource/
+model_output_matmul_readvariableop_resource0
,model_output_biasadd_readvariableop_resource
identity??model/d1/BiasAdd/ReadVariableOp?model/d1/MatMul/ReadVariableOp?model/d2/BiasAdd/ReadVariableOp?model/d2/MatMul/ReadVariableOp?model/d3/BiasAdd/ReadVariableOp?model/d3/MatMul/ReadVariableOp?#model/output/BiasAdd/ReadVariableOp?"model/output/MatMul/ReadVariableOp
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape	mhc_inputmodel/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
model/flatten/Const?
model/flatten/ReshapeReshapepeptide_inputmodel/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model/flatten/Reshapev
model/concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concat/concat/axis?
model/concat/concatConcatV2 model/flatten_1/Reshape:output:0model/flatten/Reshape:output:0!model/concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model/concat/concat?
model/d1/MatMul/ReadVariableOpReadVariableOp'model_d1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
model/d1/MatMul/ReadVariableOp?
model/d1/MatMulMatMulmodel/concat/concat:output:0&model/d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d1/MatMul?
model/d1/BiasAdd/ReadVariableOpReadVariableOp(model_d1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
model/d1/BiasAdd/ReadVariableOp?
model/d1/BiasAddBiasAddmodel/d1/MatMul:product:0'model/d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d1/BiasAddt
model/d1/ReluRelumodel/d1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/d1/Relu?
model/dropout_d1/IdentityIdentitymodel/d1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model/dropout_d1/Identityt
model/skip1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/skip1/concat/axis?
model/skip1/concatConcatV2model/concat/concat:output:0"model/dropout_d1/Identity:output:0 model/skip1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
model/skip1/concat?
model/d2/MatMul/ReadVariableOpReadVariableOp'model_d2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
model/d2/MatMul/ReadVariableOp?
model/d2/MatMulMatMulmodel/skip1/concat:output:0&model/d2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d2/MatMul?
model/d2/BiasAdd/ReadVariableOpReadVariableOp(model_d2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
model/d2/BiasAdd/ReadVariableOp?
model/d2/BiasAddBiasAddmodel/d2/MatMul:product:0'model/d2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d2/BiasAddt
model/d2/ReluRelumodel/d2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/d2/Relu?
model/dropout_d2/IdentityIdentitymodel/d2/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model/dropout_d2/Identityt
model/skip2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/skip2/concat/axis?
model/skip2/concatConcatV2model/skip1/concat:output:0"model/dropout_d2/Identity:output:0 model/skip2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????	2
model/skip2/concat?
model/d3/MatMul/ReadVariableOpReadVariableOp'model_d3_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02 
model/d3/MatMul/ReadVariableOp?
model/d3/MatMulMatMulmodel/skip2/concat:output:0&model/d3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d3/MatMul?
model/d3/BiasAdd/ReadVariableOpReadVariableOp(model_d3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
model/d3/BiasAdd/ReadVariableOp?
model/d3/BiasAddBiasAddmodel/d3/MatMul:product:0'model/d3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/d3/BiasAddt
model/d3/ReluRelumodel/d3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/d3/Relu?
model/dropout_d3/IdentityIdentitymodel/d3/Relu:activations:0*
T0*(
_output_shapes
:??????????2
model/dropout_d3/Identity?
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"model/output/MatMul/ReadVariableOp?
model/output/MatMulMatMul"model/dropout_d3/Identity:output:0*model/output/MatMul/ReadVariableOp:value:0*
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
model/output/Sigmoid?
IdentityIdentitymodel/output/Sigmoid:y:0 ^model/d1/BiasAdd/ReadVariableOp^model/d1/MatMul/ReadVariableOp ^model/d2/BiasAdd/ReadVariableOp^model/d2/MatMul/ReadVariableOp ^model/d3/BiasAdd/ReadVariableOp^model/d3/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::2B
model/d1/BiasAdd/ReadVariableOpmodel/d1/BiasAdd/ReadVariableOp2@
model/d1/MatMul/ReadVariableOpmodel/d1/MatMul/ReadVariableOp2B
model/d2/BiasAdd/ReadVariableOpmodel/d2/BiasAdd/ReadVariableOp2@
model/d2/MatMul/ReadVariableOpmodel/d2/MatMul/ReadVariableOp2B
model/d3/BiasAdd/ReadVariableOpmodel/d3/BiasAdd/ReadVariableOp2@
model/d3/MatMul/ReadVariableOpmodel/d3/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?
d
+__inference_dropout_d1_layer_call_fn_191596

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_1908972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
e
F__inference_dropout_d2_layer_call_and_return_conditional_losses_190982

inputs
identity?a
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
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
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
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqualq
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????2
dropout/mul?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/mul_1f
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_191519

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1908222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
S
'__inference_concat_layer_call_fn_191532
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_concat_layer_call_and_return_conditional_losses_1908372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
R
&__inference_skip2_layer_call_fn_191696
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip2_layer_call_and_return_conditional_losses_1910072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?I
?
A__inference_model_layer_call_and_return_conditional_losses_191117
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
identity??d1/StatefulPartitionedCall?(d1/kernel/Regularizer/Abs/ReadVariableOp?d2/StatefulPartitionedCall?(d2/kernel/Regularizer/Abs/ReadVariableOp?d3/StatefulPartitionedCall?"dropout_d1/StatefulPartitionedCall?"dropout_d2/StatefulPartitionedCall?"dropout_d3/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall	mhc_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1908082
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallpeptide_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1908222
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_concat_layer_call_and_return_conditional_losses_1908372
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_1908652
d1/StatefulPartitionedCall?
"dropout_d1/StatefulPartitionedCallStatefulPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_1908972$
"dropout_d1/StatefulPartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0+dropout_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip1_layer_call_and_return_conditional_losses_1909222
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d2_layer_call_and_return_conditional_losses_1909502
d2/StatefulPartitionedCall?
"dropout_d2/StatefulPartitionedCallStatefulPartitionedCall#d2/StatefulPartitionedCall:output:0#^dropout_d1/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d2_layer_call_and_return_conditional_losses_1909822$
"dropout_d2/StatefulPartitionedCall?
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0+dropout_d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip2_layer_call_and_return_conditional_losses_1910072
skip2/PartitionedCall?
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d3_layer_call_and_return_conditional_losses_1910272
d3/StatefulPartitionedCall?
"dropout_d3/StatefulPartitionedCallStatefulPartitionedCall#d3/StatefulPartitionedCall:output:0#^dropout_d2/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d3_layer_call_and_return_conditional_losses_1910592$
"dropout_d3/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall+dropout_d3/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1910882 
output/StatefulPartitionedCall?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_statefulpartitionedcall_args_1^d1/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d2_statefulpartitionedcall_args_1^d2/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall)^d1/kernel/Regularizer/Abs/ReadVariableOp^d2/StatefulPartitionedCall)^d2/kernel/Regularizer/Abs/ReadVariableOp^d3/StatefulPartitionedCall#^dropout_d1/StatefulPartitionedCall#^dropout_d2/StatefulPartitionedCall#^dropout_d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2H
"dropout_d1/StatefulPartitionedCall"dropout_d1/StatefulPartitionedCall2H
"dropout_d2/StatefulPartitionedCall"dropout_d2/StatefulPartitionedCall2H
"dropout_d3/StatefulPartitionedCall"dropout_d3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:) %
#
_user_specified_name	mhc_input:-)
'
_user_specified_namepeptide_input
?D
?
A__inference_model_layer_call_and_return_conditional_losses_191258

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
identity??d1/StatefulPartitionedCall?(d1/kernel/Regularizer/Abs/ReadVariableOp?d2/StatefulPartitionedCall?(d2/kernel/Regularizer/Abs/ReadVariableOp?d3/StatefulPartitionedCall?output/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1908082
flatten_1/PartitionedCall?
flatten/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1908222
flatten/PartitionedCall?
concat/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_concat_layer_call_and_return_conditional_losses_1908372
concat/PartitionedCall?
d1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0!d1_statefulpartitionedcall_args_1!d1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d1_layer_call_and_return_conditional_losses_1908652
d1/StatefulPartitionedCall?
dropout_d1/PartitionedCallPartitionedCall#d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d1_layer_call_and_return_conditional_losses_1909022
dropout_d1/PartitionedCall?
skip1/PartitionedCallPartitionedCallconcat/PartitionedCall:output:0#dropout_d1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip1_layer_call_and_return_conditional_losses_1909222
skip1/PartitionedCall?
d2/StatefulPartitionedCallStatefulPartitionedCallskip1/PartitionedCall:output:0!d2_statefulpartitionedcall_args_1!d2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d2_layer_call_and_return_conditional_losses_1909502
d2/StatefulPartitionedCall?
dropout_d2/PartitionedCallPartitionedCall#d2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d2_layer_call_and_return_conditional_losses_1909872
dropout_d2/PartitionedCall?
skip2/PartitionedCallPartitionedCallskip1/PartitionedCall:output:0#dropout_d2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????	*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_skip2_layer_call_and_return_conditional_losses_1910072
skip2/PartitionedCall?
d3/StatefulPartitionedCallStatefulPartitionedCallskip2/PartitionedCall:output:0!d3_statefulpartitionedcall_args_1!d3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d3_layer_call_and_return_conditional_losses_1910272
d3/StatefulPartitionedCall?
dropout_d3/PartitionedCallPartitionedCall#d3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d3_layer_call_and_return_conditional_losses_1910642
dropout_d3/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall#dropout_d3/PartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1910882 
output/StatefulPartitionedCall?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d1_statefulpartitionedcall_args_1^d1/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
(d2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!d2_statefulpartitionedcall_args_1^d2/StatefulPartitionedCall* 
_output_shapes
:
??*
dtype02*
(d2/kernel/Regularizer/Abs/ReadVariableOp?
d2/kernel/Regularizer/AbsAbs0d2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d2/kernel/Regularizer/Abs?
d2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d2/kernel/Regularizer/Const?
d2/kernel/Regularizer/SumSumd2/kernel/Regularizer/Abs:y:0$d2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/Sum
d2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d2/kernel/Regularizer/mul/x?
d2/kernel/Regularizer/mulMul$d2/kernel/Regularizer/mul/x:output:0"d2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/mul
d2/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d2/kernel/Regularizer/add/x?
d2/kernel/Regularizer/addAddV2$d2/kernel/Regularizer/add/x:output:0d2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d2/kernel/Regularizer/add?
IdentityIdentity'output/StatefulPartitionedCall:output:0^d1/StatefulPartitionedCall)^d1/kernel/Regularizer/Abs/ReadVariableOp^d2/StatefulPartitionedCall)^d2/kernel/Regularizer/Abs/ReadVariableOp^d3/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????":?????????::::::::28
d1/StatefulPartitionedCalld1/StatefulPartitionedCall2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp28
d2/StatefulPartitionedCalld2/StatefulPartitionedCall2T
(d2/kernel/Regularizer/Abs/ReadVariableOp(d2/kernel/Regularizer/Abs/ReadVariableOp28
d3/StatefulPartitionedCalld3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_190822

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
>__inference_d1_layer_call_and_return_conditional_losses_191559

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?(d1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp)^d1/kernel/Regularizer/Abs/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp:& "
 
_user_specified_nameinputs
?	
?
>__inference_d3_layer_call_and_return_conditional_losses_191707

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
>__inference_d1_layer_call_and_return_conditional_losses_190865

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?(d1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
(d1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp* 
_output_shapes
:
??*
dtype02*
(d1/kernel/Regularizer/Abs/ReadVariableOp?
d1/kernel/Regularizer/AbsAbs0d1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
d1/kernel/Regularizer/Abs?
d1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
d1/kernel/Regularizer/Const?
d1/kernel/Regularizer/SumSumd1/kernel/Regularizer/Abs:y:0$d1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/Sum
d1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
d1/kernel/Regularizer/mul/x?
d1/kernel/Regularizer/mulMul$d1/kernel/Regularizer/mul/x:output:0"d1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/mul
d1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
d1/kernel/Regularizer/add/x?
d1/kernel/Regularizer/addAddV2$d1/kernel/Regularizer/add/x:output:0d1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
d1/kernel/Regularizer/add?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp)^d1/kernel/Regularizer/Abs/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2T
(d1/kernel/Regularizer/Abs/ReadVariableOp(d1/kernel/Regularizer/Abs/ReadVariableOp:& "
 
_user_specified_nameinputs
?
d
F__inference_dropout_d2_layer_call_and_return_conditional_losses_190987

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
l
B__inference_concat_layer_call_and_return_conditional_losses_190837

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?	
?
>__inference_d3_layer_call_and_return_conditional_losses_191027

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
d
+__inference_dropout_d2_layer_call_fn_191678

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d2_layer_call_and_return_conditional_losses_1909822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
B__inference_output_layer_call_and_return_conditional_losses_191760

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
m
A__inference_skip1_layer_call_and_return_conditional_losses_191608
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
 
cond_true_191820
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
?
F
*__inference_flatten_1_layer_call_fn_191508

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1908082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????":& "
 
_user_specified_nameinputs
?
d
+__inference_dropout_d3_layer_call_fn_191744

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d3_layer_call_and_return_conditional_losses_1910592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
n
B__inference_concat_layer_call_and_return_conditional_losses_191526
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:??????????:??????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?"
?
__inference__traced_save_191852
file_prefix(
$savev2_d1_kernel_read_readvariableop&
"savev2_d1_bias_read_readvariableop(
$savev2_d2_kernel_read_readvariableop&
"savev2_d2_bias_read_readvariableop(
$savev2_d3_kernel_read_readvariableop&
"savev2_d3_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:0*
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatch?
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
cond_false_191821*
output_shapes
: *#
then_branchR
cond_true_1918202
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_d1_kernel_read_readvariableop"savev2_d1_bias_read_readvariableop$savev2_d2_kernel_read_readvariableop"savev2_d2_bias_read_readvariableop$savev2_d3_kernel_read_readvariableop"savev2_d3_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
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

identity_1Identity_1:output:0*a
_input_shapesP
N: :
??:?:
??:?:
?	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
G
+__inference_dropout_d2_layer_call_fn_191683

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_d2_layer_call_and_return_conditional_losses_1909872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
?
#__inference_d3_layer_call_fn_191714

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_d3_layer_call_and_return_conditional_losses_1910272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
	mhc_input6
serving_default_mhc_input:0?????????"
K
peptide_input:
serving_default_peptide_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ƹ
?N
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
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?J
_tf_keras_model?J{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}, "name": "mhc_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}, "name": "peptide_input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mhc_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["peptide_input", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concat", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip1", "inbound_nodes": [[["concat", 0, 0, {}], ["dropout_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d2", "inbound_nodes": [[["skip1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d2", "inbound_nodes": [[["d2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip2", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip2", "inbound_nodes": [[["skip1", 0, 0, {}], ["dropout_d2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d3", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d3", "inbound_nodes": [[["skip2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d3", "inbound_nodes": [[["d3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d3", 0, 0, {}]]]}], "input_layers": [["mhc_input", 0, 0], ["peptide_input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}, "name": "mhc_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}, "name": "peptide_input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mhc_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["peptide_input", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concat", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d1", "inbound_nodes": [[["concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d1", "inbound_nodes": [[["d1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip1", "inbound_nodes": [[["concat", 0, 0, {}], ["dropout_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d2", "inbound_nodes": [[["skip1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d2", "inbound_nodes": [[["d2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "skip2", "trainable": true, "dtype": "float32", "axis": 1}, "name": "skip2", "inbound_nodes": [[["skip1", 0, 0, {}], ["dropout_d2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "d3", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "d3", "inbound_nodes": [[["skip2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_d3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_d3", "inbound_nodes": [[["d3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_d3", 0, 0, {}]]]}], "input_layers": [["mhc_input", 0, 0], ["peptide_input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "loss", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0011339303598057013, "decay": 0.0, "rho": 0.9, "momentum": 0.5, "epsilon": 6.848580326162904e-07, "centered": true}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "mhc_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 34, 20], "config": {"batch_input_shape": [null, 34, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "mhc_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "peptide_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 15, 20], "config": {"batch_input_shape": [null, 15, 20], "dtype": "float32", "sparse": false, "ragged": false, "name": "peptide_input"}}
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
regularization_losses
trainable_variables
	variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concat", "trainable": true, "dtype": "float32", "axis": 1}}
?

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 980}}}}
?
'regularization_losses
(trainable_variables
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_d1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
+regularization_losses
,trainable_variables
-	variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "skip1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "skip1", "trainable": true, "dtype": "float32", "axis": 1}}
?

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "d2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.0, "l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1108}}}}
?
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_d2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "skip2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "skip2", "trainable": true, "dtype": "float32", "axis": 1}}
?

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "d3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "d3", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1236}}}}
?
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_d3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_d3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Gkernel
Hbias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
"
	optimizer
0
?0
?1"
trackable_list_wrapper
X
!0
"1
/2
03
=4
>5
G6
H7"
trackable_list_wrapper
X
!0
"1
/2
03
=4
>5
G6
H7"
trackable_list_wrapper
?
regularization_losses
Mmetrics

Nlayers
Olayer_regularization_losses
Pnon_trainable_variables
trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Qmetrics

Rlayers
Slayer_regularization_losses
Tnon_trainable_variables
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Umetrics

Vlayers
Wlayer_regularization_losses
Xnon_trainable_variables
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
Ymetrics

Zlayers
[layer_regularization_losses
\non_trainable_variables
trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2	d1/kernel
:?2d1/bias
(
?0"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
#regularization_losses
]metrics

^layers
_layer_regularization_losses
`non_trainable_variables
$trainable_variables
%	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'regularization_losses
ametrics

blayers
clayer_regularization_losses
dnon_trainable_variables
(trainable_variables
)	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+regularization_losses
emetrics

flayers
glayer_regularization_losses
hnon_trainable_variables
,trainable_variables
-	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2	d2/kernel
:?2d2/bias
(
?0"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
1regularization_losses
imetrics

jlayers
klayer_regularization_losses
lnon_trainable_variables
2trainable_variables
3	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5regularization_losses
mmetrics

nlayers
olayer_regularization_losses
pnon_trainable_variables
6trainable_variables
7	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
9regularization_losses
qmetrics

rlayers
slayer_regularization_losses
tnon_trainable_variables
:trainable_variables
;	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
?	?2	d3/kernel
:?2d3/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?regularization_losses
umetrics

vlayers
wlayer_regularization_losses
xnon_trainable_variables
@trainable_variables
A	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cregularization_losses
ymetrics

zlayers
{layer_regularization_losses
|non_trainable_variables
Dtrainable_variables
E	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
Iregularization_losses
}metrics

~layers
layer_regularization_losses
?non_trainable_variables
Jtrainable_variables
K	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
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
(
?0"
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
?0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
!__inference__wrapped_model_190797?
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
annotations? *^?[
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
?2?
&__inference_model_layer_call_fn_191483
&__inference_model_layer_call_fn_191497
&__inference_model_layer_call_fn_191269
&__inference_model_layer_call_fn_191214?
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
?2?
A__inference_model_layer_call_and_return_conditional_losses_191158
A__inference_model_layer_call_and_return_conditional_losses_191117
A__inference_model_layer_call_and_return_conditional_losses_191407
A__inference_model_layer_call_and_return_conditional_losses_191469?
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
*__inference_flatten_1_layer_call_fn_191508?
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_191503?
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
(__inference_flatten_layer_call_fn_191519?
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
C__inference_flatten_layer_call_and_return_conditional_losses_191514?
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
'__inference_concat_layer_call_fn_191532?
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
B__inference_concat_layer_call_and_return_conditional_losses_191526?
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
#__inference_d1_layer_call_fn_191566?
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
>__inference_d1_layer_call_and_return_conditional_losses_191559?
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
+__inference_dropout_d1_layer_call_fn_191596
+__inference_dropout_d1_layer_call_fn_191601?
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
F__inference_dropout_d1_layer_call_and_return_conditional_losses_191586
F__inference_dropout_d1_layer_call_and_return_conditional_losses_191591?
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
&__inference_skip1_layer_call_fn_191614?
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
A__inference_skip1_layer_call_and_return_conditional_losses_191608?
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
#__inference_d2_layer_call_fn_191648?
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
>__inference_d2_layer_call_and_return_conditional_losses_191641?
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
+__inference_dropout_d2_layer_call_fn_191678
+__inference_dropout_d2_layer_call_fn_191683?
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
F__inference_dropout_d2_layer_call_and_return_conditional_losses_191668
F__inference_dropout_d2_layer_call_and_return_conditional_losses_191673?
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
&__inference_skip2_layer_call_fn_191696?
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
A__inference_skip2_layer_call_and_return_conditional_losses_191690?
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
#__inference_d3_layer_call_fn_191714?
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
>__inference_d3_layer_call_and_return_conditional_losses_191707?
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
+__inference_dropout_d3_layer_call_fn_191744
+__inference_dropout_d3_layer_call_fn_191749?
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
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191734
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191739?
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
'__inference_output_layer_call_fn_191767?
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
B__inference_output_layer_call_and_return_conditional_losses_191760?
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
__inference_loss_fn_0_191780?
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
?2?
__inference_loss_fn_1_191793?
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
BB@
$__inference_signature_wrapper_191300	mhc_inputpeptide_input?
!__inference__wrapped_model_190797?!"/0=>GHh?e
^?[
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
? "/?,
*
output ?
output??????????
B__inference_concat_layer_call_and_return_conditional_losses_191526?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
'__inference_concat_layer_call_fn_191532y\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "????????????
>__inference_d1_layer_call_and_return_conditional_losses_191559^!"0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? x
#__inference_d1_layer_call_fn_191566Q!"0?-
&?#
!?
inputs??????????
? "????????????
>__inference_d2_layer_call_and_return_conditional_losses_191641^/00?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? x
#__inference_d2_layer_call_fn_191648Q/00?-
&?#
!?
inputs??????????
? "????????????
>__inference_d3_layer_call_and_return_conditional_losses_191707^=>0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? x
#__inference_d3_layer_call_fn_191714Q=>0?-
&?#
!?
inputs??????????	
? "????????????
F__inference_dropout_d1_layer_call_and_return_conditional_losses_191586^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
F__inference_dropout_d1_layer_call_and_return_conditional_losses_191591^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
+__inference_dropout_d1_layer_call_fn_191596Q4?1
*?'
!?
inputs??????????
p
? "????????????
+__inference_dropout_d1_layer_call_fn_191601Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_dropout_d2_layer_call_and_return_conditional_losses_191668^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
F__inference_dropout_d2_layer_call_and_return_conditional_losses_191673^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
+__inference_dropout_d2_layer_call_fn_191678Q4?1
*?'
!?
inputs??????????
p
? "????????????
+__inference_dropout_d2_layer_call_fn_191683Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191734^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
F__inference_dropout_d3_layer_call_and_return_conditional_losses_191739^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
+__inference_dropout_d3_layer_call_fn_191744Q4?1
*?'
!?
inputs??????????
p
? "????????????
+__inference_dropout_d3_layer_call_fn_191749Q4?1
*?'
!?
inputs??????????
p 
? "????????????
E__inference_flatten_1_layer_call_and_return_conditional_losses_191503]3?0
)?&
$?!
inputs?????????"
? "&?#
?
0??????????
? ~
*__inference_flatten_1_layer_call_fn_191508P3?0
)?&
$?!
inputs?????????"
? "????????????
C__inference_flatten_layer_call_and_return_conditional_losses_191514]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? |
(__inference_flatten_layer_call_fn_191519P3?0
)?&
$?!
inputs?????????
? "???????????;
__inference_loss_fn_0_191780!?

? 
? "? ;
__inference_loss_fn_1_191793/?

? 
? "? ?
A__inference_model_layer_call_and_return_conditional_losses_191117?!"/0=>GHp?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_191158?!"/0=>GHp?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_191407?!"/0=>GHj?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_191469?!"/0=>GHj?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_191214?!"/0=>GHp?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p

 
? "???????????
&__inference_model_layer_call_fn_191269?!"/0=>GHp?m
f?c
Y?V
'?$
	mhc_input?????????"
+?(
peptide_input?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_191483?!"/0=>GHj?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p

 
? "???????????
&__inference_model_layer_call_fn_191497?!"/0=>GHj?g
`?]
S?P
&?#
inputs/0?????????"
&?#
inputs/1?????????
p 

 
? "???????????
B__inference_output_layer_call_and_return_conditional_losses_191760]GH0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_output_layer_call_fn_191767PGH0?-
&?#
!?
inputs??????????
? "???????????
$__inference_signature_wrapper_191300?!"/0=>GH??~
? 
w?t
4
	mhc_input'?$
	mhc_input?????????"
<
peptide_input+?(
peptide_input?????????"/?,
*
output ?
output??????????
A__inference_skip1_layer_call_and_return_conditional_losses_191608?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????
? ?
&__inference_skip1_layer_call_fn_191614y\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "????????????
A__inference_skip2_layer_call_and_return_conditional_losses_191690?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "&?#
?
0??????????	
? ?
&__inference_skip2_layer_call_fn_191696y\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "???????????	