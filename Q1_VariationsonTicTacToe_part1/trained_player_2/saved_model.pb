؇
��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
$
DisableCopyOnRead
resource�
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
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
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
�
QNetwork/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Q*'
shared_nameQNetwork/dense_11/bias
}
*QNetwork/dense_11/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_11/bias*
_output_shapes
:Q*
dtype0
�
QNetwork/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<Q*)
shared_nameQNetwork/dense_11/kernel
�
,QNetwork/dense_11/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_11/kernel*
_output_shapes

:<Q*
dtype0
�
&QNetwork/EncodingNetwork/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*7
shared_name(&QNetwork/EncodingNetwork/dense_10/bias
�
:QNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_10/bias*
_output_shapes
:<*
dtype0
�
(QNetwork/EncodingNetwork/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P<*9
shared_name*(QNetwork/EncodingNetwork/dense_10/kernel
�
<QNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_10/kernel*
_output_shapes

:P<*
dtype0
�
%QNetwork/EncodingNetwork/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%QNetwork/EncodingNetwork/dense_9/bias
�
9QNetwork/EncodingNetwork/dense_9/bias/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense_9/bias*
_output_shapes
:P*
dtype0
�
'QNetwork/EncodingNetwork/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dP*8
shared_name)'QNetwork/EncodingNetwork/dense_9/kernel
�
;QNetwork/EncodingNetwork/dense_9/kernel/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/dense_9/kernel*
_output_shapes

:dP*
dtype0
�
%QNetwork/EncodingNetwork/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*6
shared_name'%QNetwork/EncodingNetwork/dense_8/bias
�
9QNetwork/EncodingNetwork/dense_8/bias/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense_8/bias*
_output_shapes
:d*
dtype0
�
'QNetwork/EncodingNetwork/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Qd*8
shared_name)'QNetwork/EncodingNetwork/dense_8/kernel
�
;QNetwork/EncodingNetwork/dense_8/kernel/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/dense_8/kernel*
_output_shapes

:Qd*
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
l
action_0_discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������

action_0_observationPlaceholder*+
_output_shapes
:���������		*
dtype0* 
shape:���������		
j
action_0_rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0_step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type'QNetwork/EncodingNetwork/dense_8/kernel%QNetwork/EncodingNetwork/dense_8/bias'QNetwork/EncodingNetwork/dense_9/kernel%QNetwork/EncodingNetwork/dense_9/bias(QNetwork/EncodingNetwork/dense_10/kernel&QNetwork/EncodingNetwork/dense_10/biasQNetwork/dense_11/kernelQNetwork/dense_11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_signature_wrapper_function_with_signature_146589204
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_signature_wrapper_function_with_signature_146589214
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_signature_wrapper_function_with_signature_146589232
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_signature_wrapper_function_with_signature_146589227

NoOpNoOp
�)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�(
value�(B�( B�(
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures*
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
1
2
3
4
5
6
7*

_wrapped_policy*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

action
get_initial_state
get_train_step
get_metadata* 
mg
VARIABLE_VALUE'QNetwork/EncodingNetwork/dense_8/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense_8/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE'QNetwork/EncodingNetwork/dense_9/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense_9/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_10/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_10/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEQNetwork/dense_11/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEQNetwork/dense_11/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*


_q_network*
* 
* 
* 
* 
* 
* 
* 
* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_encoder
$_q_value_layer*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_postprocessing_layers*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias*
* 

#0
$1*
* 
* 
* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
 
<0
=1
>2
?3*

0
1*

0
1*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
* 
 
<0
=1
>2
?3*
* 
* 
* 
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

kernel
bias*
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

kernel
bias*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

kernel
bias*
* 
* 
* 
* 
* 
* 
* 
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable'QNetwork/EncodingNetwork/dense_8/kernel%QNetwork/EncodingNetwork/dense_8/bias'QNetwork/EncodingNetwork/dense_9/kernel%QNetwork/EncodingNetwork/dense_9/bias(QNetwork/EncodingNetwork/dense_10/kernel&QNetwork/EncodingNetwork/dense_10/biasQNetwork/dense_11/kernelQNetwork/dense_11/biasConst*
Tin
2*
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
GPU 2J 8� *+
f&R$
"__inference__traced_save_146589498
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable'QNetwork/EncodingNetwork/dense_8/kernel%QNetwork/EncodingNetwork/dense_8/bias'QNetwork/EncodingNetwork/dense_9/kernel%QNetwork/EncodingNetwork/dense_9/bias(QNetwork/EncodingNetwork/dense_10/kernel&QNetwork/EncodingNetwork/dense_10/biasQNetwork/dense_11/kernelQNetwork/dense_11/bias*
Tin
2
*
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
GPU 2J 8� *.
f)R'
%__inference__traced_restore_146589534��
�\
�	
+__inference_polymorphic_action_fn_146589301
	step_type

reward
discount
observationQ
?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:QdN
@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource:dQ
?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource:dPN
@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource:PR
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:P<O
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:<B
0qnetwork_dense_11_matmul_readvariableop_resource:<Q?
1qnetwork_dense_11_biasadd_readvariableop_resource:Q
identity��8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp�(QNetwork/dense_11/BiasAdd/ReadVariableOp�'QNetwork/dense_11/MatMul/ReadVariableOpy
(QNetwork/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Q   �
*QNetwork/EncodingNetwork/flatten_2/ReshapeReshapeobservation1QNetwork/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������Q�
%QNetwork/EncodingNetwork/dense_8/CastCast3QNetwork/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������Q�
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:Qd*
dtype0�
'QNetwork/EncodingNetwork/dense_8/MatMulMatMul)QNetwork/EncodingNetwork/dense_8/Cast:y:0>QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
(QNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_8/MatMul:product:0?QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
%QNetwork/EncodingNetwork/dense_8/ReluRelu1QNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype0�
'QNetwork/EncodingNetwork/dense_9/MatMulMatMul3QNetwork/EncodingNetwork/dense_8/Relu:activations:0>QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
(QNetwork/EncodingNetwork/dense_9/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_9/MatMul:product:0?QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%QNetwork/EncodingNetwork/dense_9/ReluRelu1QNetwork/EncodingNetwork/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype0�
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/dense_9/Relu:activations:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������<�
'QNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_11_matmul_readvariableop_resource*
_output_shapes

:<Q*
dtype0�
QNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0/QNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q�
(QNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype0�
QNetwork/dense_11/BiasAddBiasAdd"QNetwork/dense_11/MatMul:product:00QNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Ql
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax"QNetwork/dense_11/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB q
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:����������
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :P�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:����������
NoOpNoOp9^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp)^QNetwork/dense_11/BiasAdd/ReadVariableOp(^QNetwork/dense_11/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������		: : : : : : : : 2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp2T
(QNetwork/dense_11/BiasAdd/ReadVariableOp(QNetwork/dense_11/BiasAdd/ReadVariableOp2R
'QNetwork/dense_11/MatMul/ReadVariableOp'QNetwork/dense_11/MatMul/ReadVariableOp:($
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
resource:XT
+
_output_shapes
:���������		
%
_user_specified_nameobservation:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:KG
#
_output_shapes
:���������
 
_user_specified_namereward:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type
�

?__inference_signature_wrapper_function_with_signature_146589227
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *6
f1R/
-__inference_function_with_signature_146589220^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	146589223
�
9
'__inference_get_initial_state_146589417

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�@
�	
1__inference_polymorphic_distribution_fn_146589414
	step_type

reward
discount
observationQ
?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:QdN
@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource:dQ
?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource:dPN
@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource:PR
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:P<O
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:<B
0qnetwork_dense_11_matmul_readvariableop_resource:<Q?
1qnetwork_dense_11_biasadd_readvariableop_resource:Q
identity

identity_1

identity_2��8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp�(QNetwork/dense_11/BiasAdd/ReadVariableOp�'QNetwork/dense_11/MatMul/ReadVariableOpy
(QNetwork/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Q   �
*QNetwork/EncodingNetwork/flatten_2/ReshapeReshapeobservation1QNetwork/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������Q�
%QNetwork/EncodingNetwork/dense_8/CastCast3QNetwork/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������Q�
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:Qd*
dtype0�
'QNetwork/EncodingNetwork/dense_8/MatMulMatMul)QNetwork/EncodingNetwork/dense_8/Cast:y:0>QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
(QNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_8/MatMul:product:0?QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
%QNetwork/EncodingNetwork/dense_8/ReluRelu1QNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype0�
'QNetwork/EncodingNetwork/dense_9/MatMulMatMul3QNetwork/EncodingNetwork/dense_8/Relu:activations:0>QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
(QNetwork/EncodingNetwork/dense_9/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_9/MatMul:product:0?QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%QNetwork/EncodingNetwork/dense_9/ReluRelu1QNetwork/EncodingNetwork/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype0�
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/dense_9/Relu:activations:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������<�
'QNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_11_matmul_readvariableop_resource*
_output_shapes

:<Q*
dtype0�
QNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0/QNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q�
(QNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype0�
QNetwork/dense_11/BiasAddBiasAdd"QNetwork/dense_11/MatMul:product:00QNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Ql
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax"QNetwork/dense_11/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: f

Identity_1IdentityCategorical/mode/Cast:y:0^NoOp*
T0*#
_output_shapes
:���������[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp9^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp)^QNetwork/dense_11/BiasAdd/ReadVariableOp(^QNetwork/dense_11/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������		: : : : : : : : 2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp2T
(QNetwork/dense_11/BiasAdd/ReadVariableOp(QNetwork/dense_11/BiasAdd/ReadVariableOp2R
'QNetwork/dense_11/MatMul/ReadVariableOp'QNetwork/dense_11/MatMul/ReadVariableOp:($
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
resource:XT
+
_output_shapes
:���������		
%
_user_specified_nameobservation:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:KG
#
_output_shapes
:���������
 
_user_specified_namereward:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type
_
 
__inference_<lambda>_146588473*(
_construction_contextkEagerRuntime*
_input_shapes 
�
A
?__inference_signature_wrapper_function_with_signature_146589232�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *6
f1R/
-__inference_function_with_signature_146589229*(
_construction_contextkEagerRuntime*
_input_shapes 
�
m
-__inference_function_with_signature_146589220
unknown:	 
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference_<lambda>_146588471^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	146589216
�T
�	
"__inference__traced_save_146589498
file_prefix)
read_disablecopyonread_variable:	 R
@read_1_disablecopyonread_qnetwork_encodingnetwork_dense_8_kernel:QdL
>read_2_disablecopyonread_qnetwork_encodingnetwork_dense_8_bias:dR
@read_3_disablecopyonread_qnetwork_encodingnetwork_dense_9_kernel:dPL
>read_4_disablecopyonread_qnetwork_encodingnetwork_dense_9_bias:PS
Aread_5_disablecopyonread_qnetwork_encodingnetwork_dense_10_kernel:P<M
?read_6_disablecopyonread_qnetwork_encodingnetwork_dense_10_bias:<C
1read_7_disablecopyonread_qnetwork_dense_11_kernel:<Q=
/read_8_disablecopyonread_qnetwork_dense_11_bias:Q
savev2_const
identity_19��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOpw
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
: q
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_variable^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead@read_1_disablecopyonread_qnetwork_encodingnetwork_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp@read_1_disablecopyonread_qnetwork_encodingnetwork_dense_8_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:Qd*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:Qdc

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:Qd�
Read_2/DisableCopyOnReadDisableCopyOnRead>read_2_disablecopyonread_qnetwork_encodingnetwork_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp>read_2_disablecopyonread_qnetwork_encodingnetwork_dense_8_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:d*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:d_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:d�
Read_3/DisableCopyOnReadDisableCopyOnRead@read_3_disablecopyonread_qnetwork_encodingnetwork_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp@read_3_disablecopyonread_qnetwork_encodingnetwork_dense_9_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:dP*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:dPc

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:dP�
Read_4/DisableCopyOnReadDisableCopyOnRead>read_4_disablecopyonread_qnetwork_encodingnetwork_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp>read_4_disablecopyonread_qnetwork_encodingnetwork_dense_9_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:P*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:P_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:P�
Read_5/DisableCopyOnReadDisableCopyOnReadAread_5_disablecopyonread_qnetwork_encodingnetwork_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpAread_5_disablecopyonread_qnetwork_encodingnetwork_dense_10_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:P<*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:P<e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:P<�
Read_6/DisableCopyOnReadDisableCopyOnRead?read_6_disablecopyonread_qnetwork_encodingnetwork_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp?read_6_disablecopyonread_qnetwork_encodingnetwork_dense_10_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:<*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:<a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:<�
Read_7/DisableCopyOnReadDisableCopyOnRead1read_7_disablecopyonread_qnetwork_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp1read_7_disablecopyonread_qnetwork_dense_11_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:<Q*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:<Qe
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:<Q�
Read_8/DisableCopyOnReadDisableCopyOnRead/read_8_disablecopyonread_qnetwork_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp/read_8_disablecopyonread_qnetwork_dense_11_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:Q*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Qa
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:Q�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*�
value�B�
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_18Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_19IdentityIdentity_18:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp*
_output_shapes
 "#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp:=
9

_output_shapes
: 

_user_specified_nameConst:6	2
0
_user_specified_nameQNetwork/dense_11/bias:84
2
_user_specified_nameQNetwork/dense_11/kernel:FB
@
_user_specified_name(&QNetwork/EncodingNetwork/dense_10/bias:HD
B
_user_specified_name*(QNetwork/EncodingNetwork/dense_10/kernel:EA
?
_user_specified_name'%QNetwork/EncodingNetwork/dense_9/bias:GC
A
_user_specified_name)'QNetwork/EncodingNetwork/dense_9/kernel:EA
?
_user_specified_name'%QNetwork/EncodingNetwork/dense_8/bias:GC
A
_user_specified_name)'QNetwork/EncodingNetwork/dense_8/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�]
�	
+__inference_polymorphic_action_fn_146589370
time_step_step_type
time_step_reward
time_step_discount
time_step_observationQ
?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:QdN
@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource:dQ
?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource:dPN
@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource:PR
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:P<O
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:<B
0qnetwork_dense_11_matmul_readvariableop_resource:<Q?
1qnetwork_dense_11_biasadd_readvariableop_resource:Q
identity��8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp�(QNetwork/dense_11/BiasAdd/ReadVariableOp�'QNetwork/dense_11/MatMul/ReadVariableOpy
(QNetwork/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Q   �
*QNetwork/EncodingNetwork/flatten_2/ReshapeReshapetime_step_observation1QNetwork/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������Q�
%QNetwork/EncodingNetwork/dense_8/CastCast3QNetwork/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������Q�
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:Qd*
dtype0�
'QNetwork/EncodingNetwork/dense_8/MatMulMatMul)QNetwork/EncodingNetwork/dense_8/Cast:y:0>QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
(QNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_8/MatMul:product:0?QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
%QNetwork/EncodingNetwork/dense_8/ReluRelu1QNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype0�
'QNetwork/EncodingNetwork/dense_9/MatMulMatMul3QNetwork/EncodingNetwork/dense_8/Relu:activations:0>QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
(QNetwork/EncodingNetwork/dense_9/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_9/MatMul:product:0?QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%QNetwork/EncodingNetwork/dense_9/ReluRelu1QNetwork/EncodingNetwork/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype0�
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/dense_9/Relu:activations:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������<�
'QNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_11_matmul_readvariableop_resource*
_output_shapes

:<Q*
dtype0�
QNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0/QNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q�
(QNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype0�
QNetwork/dense_11/BiasAddBiasAdd"QNetwork/dense_11/MatMul:product:00QNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Ql
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax"QNetwork/dense_11/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB q
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:����������
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :P�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:����������
NoOpNoOp9^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp)^QNetwork/dense_11/BiasAdd/ReadVariableOp(^QNetwork/dense_11/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������		: : : : : : : : 2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp2T
(QNetwork/dense_11/BiasAdd/ReadVariableOp(QNetwork/dense_11/BiasAdd/ReadVariableOp2R
'QNetwork/dense_11/MatMul/ReadVariableOp'QNetwork/dense_11/MatMul/ReadVariableOp:($
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
resource:b^
+
_output_shapes
:���������		
/
_user_specified_nametime_step_observation:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step_discount:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step_reward:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step_step_type
�
�
?__inference_signature_wrapper_function_with_signature_146589204
discount
observation

reward
	step_type
unknown:Qd
	unknown_0:d
	unknown_1:dP
	unknown_2:P
	unknown_3:P<
	unknown_4:<
	unknown_5:<Q
	unknown_6:Q
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *6
f1R/
-__inference_function_with_signature_146589179k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������		:���������:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:)%
#
_user_specified_name	146589200:)
%
#
_user_specified_name	146589198:)	%
#
_user_specified_name	146589196:)%
#
_user_specified_name	146589194:)%
#
_user_specified_name	146589192:)%
#
_user_specified_name	146589190:)%
#
_user_specified_name	146589188:)%
#
_user_specified_name	146589186:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:ZV
+
_output_shapes
:���������		
'
_user_specified_name0/observation:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount
�]
�	
+__inference_polymorphic_action_fn_146589160
	time_step
time_step_1
time_step_2
time_step_3Q
?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource:QdN
@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource:dQ
?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource:dPN
@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource:PR
@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:P<O
Aqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:<B
0qnetwork_dense_11_matmul_readvariableop_resource:<Q?
1qnetwork_dense_11_biasadd_readvariableop_resource:Q
identity��8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp�7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp�6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp�(QNetwork/dense_11/BiasAdd/ReadVariableOp�'QNetwork/dense_11/MatMul/ReadVariableOpy
(QNetwork/EncodingNetwork/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����Q   �
*QNetwork/EncodingNetwork/flatten_2/ReshapeReshapetime_step_31QNetwork/EncodingNetwork/flatten_2/Const:output:0*
T0*'
_output_shapes
:���������Q�
%QNetwork/EncodingNetwork/dense_8/CastCast3QNetwork/EncodingNetwork/flatten_2/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:���������Q�
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_8_matmul_readvariableop_resource*
_output_shapes

:Qd*
dtype0�
'QNetwork/EncodingNetwork/dense_8/MatMulMatMul)QNetwork/EncodingNetwork/dense_8/Cast:y:0>QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
(QNetwork/EncodingNetwork/dense_8/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_8/MatMul:product:0?QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
%QNetwork/EncodingNetwork/dense_8/ReluRelu1QNetwork/EncodingNetwork/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOpReadVariableOp?qnetwork_encodingnetwork_dense_9_matmul_readvariableop_resource*
_output_shapes

:dP*
dtype0�
'QNetwork/EncodingNetwork/dense_9/MatMulMatMul3QNetwork/EncodingNetwork/dense_8/Relu:activations:0>QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_9_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
(QNetwork/EncodingNetwork/dense_9/BiasAddBiasAdd1QNetwork/EncodingNetwork/dense_9/MatMul:product:0?QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%QNetwork/EncodingNetwork/dense_9/ReluRelu1QNetwork/EncodingNetwork/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes

:P<*
dtype0�
(QNetwork/EncodingNetwork/dense_10/MatMulMatMul3QNetwork/EncodingNetwork/dense_9/Relu:activations:0?QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0�
)QNetwork/EncodingNetwork/dense_10/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_10/MatMul:product:0@QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<�
&QNetwork/EncodingNetwork/dense_10/ReluRelu2QNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������<�
'QNetwork/dense_11/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_11_matmul_readvariableop_resource*
_output_shapes

:<Q*
dtype0�
QNetwork/dense_11/MatMulMatMul4QNetwork/EncodingNetwork/dense_10/Relu:activations:0/QNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Q�
(QNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype0�
QNetwork/dense_11/BiasAddBiasAdd"QNetwork/dense_11/MatMul:product:00QNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Ql
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
����������
Categorical/mode/ArgMaxArgMax"QNetwork/dense_11/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������T
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB q
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
::��\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:����������
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::��t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:���������Y
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :P�
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������Q
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:����������
NoOpNoOp9^QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp8^QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7^QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp)^QNetwork/dense_11/BiasAdd/ReadVariableOp(^QNetwork/dense_11/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������		: : : : : : : : 2t
8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_8/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_8/MatMul/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp7QNetwork/EncodingNetwork/dense_9/BiasAdd/ReadVariableOp2p
6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp6QNetwork/EncodingNetwork/dense_9/MatMul/ReadVariableOp2T
(QNetwork/dense_11/BiasAdd/ReadVariableOp(QNetwork/dense_11/BiasAdd/ReadVariableOp2R
'QNetwork/dense_11/MatMul/ReadVariableOp'QNetwork/dense_11/MatMul/ReadVariableOp:($
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
resource:VR
+
_output_shapes
:���������		
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step
�
?
-__inference_function_with_signature_146589210

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *0
f+R)
'__inference_get_initial_state_146589209*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
/
-__inference_function_with_signature_146589229�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference_<lambda>_146588473*(
_construction_contextkEagerRuntime*
_input_shapes 
�
e
__inference_<lambda>_146588471!
readvariableop_resource:	 
identity	��ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: 3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp:( $
"
_user_specified_name
resource
�
Q
?__inference_signature_wrapper_function_with_signature_146589214

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *6
f1R/
-__inference_function_with_signature_146589210*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
9
'__inference_get_initial_state_146589209

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
�
-__inference_function_with_signature_146589179
	step_type

reward
discount
observation
unknown:Qd
	unknown_0:d
	unknown_1:dP
	unknown_2:P
	unknown_3:P<
	unknown_4:<
	unknown_5:<Q
	unknown_6:Q
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *4
f/R-
+__inference_polymorphic_action_fn_146589160k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������		: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:)%
#
_user_specified_name	146589175:)
%
#
_user_specified_name	146589173:)	%
#
_user_specified_name	146589171:)%
#
_user_specified_name	146589169:)%
#
_user_specified_name	146589167:)%
#
_user_specified_name	146589165:)%
#
_user_specified_name	146589163:)%
#
_user_specified_name	146589161:ZV
+
_output_shapes
:���������		
'
_user_specified_name0/observation:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type
�1
�
%__inference__traced_restore_146589534
file_prefix#
assignvariableop_variable:	 L
:assignvariableop_1_qnetwork_encodingnetwork_dense_8_kernel:QdF
8assignvariableop_2_qnetwork_encodingnetwork_dense_8_bias:dL
:assignvariableop_3_qnetwork_encodingnetwork_dense_9_kernel:dPF
8assignvariableop_4_qnetwork_encodingnetwork_dense_9_bias:PM
;assignvariableop_5_qnetwork_encodingnetwork_dense_10_kernel:P<G
9assignvariableop_6_qnetwork_encodingnetwork_dense_10_bias:<=
+assignvariableop_7_qnetwork_dense_11_kernel:<Q7
)assignvariableop_8_qnetwork_dense_11_bias:Q
identity_10��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*�
value�B�
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp:assignvariableop_1_qnetwork_encodingnetwork_dense_8_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp8assignvariableop_2_qnetwork_encodingnetwork_dense_8_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_qnetwork_encodingnetwork_dense_9_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp8assignvariableop_4_qnetwork_encodingnetwork_dense_9_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp;assignvariableop_5_qnetwork_encodingnetwork_dense_10_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp9assignvariableop_6_qnetwork_encodingnetwork_dense_10_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_qnetwork_dense_11_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_qnetwork_dense_11_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: V
Identity_10IdentityIdentity_9:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8*
_output_shapes
 "#
identity_10Identity_10:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82$
AssignVariableOpAssignVariableOp:6	2
0
_user_specified_nameQNetwork/dense_11/bias:84
2
_user_specified_nameQNetwork/dense_11/kernel:FB
@
_user_specified_name(&QNetwork/EncodingNetwork/dense_10/bias:HD
B
_user_specified_name*(QNetwork/EncodingNetwork/dense_10/kernel:EA
?
_user_specified_name'%QNetwork/EncodingNetwork/dense_9/bias:GC
A
_user_specified_name)'QNetwork/EncodingNetwork/dense_9/kernel:EA
?
_user_specified_name'%QNetwork/EncodingNetwork/dense_8/bias:GC
A
_user_specified_name)'QNetwork/EncodingNetwork/dense_8/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0_discount:0���������
B
0/observation1
action_0_observation:0���������		
0
0/reward$
action_0_reward:0���������
6
0/step_type'
action_0_step_type:0���������6
action,
StatefulPartitionedCall:0���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:�s
�

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
Y
0
1
2
3
4
5
6
7"
trackable_tuple_wrapper
5
_wrapped_policy"
trackable_dict_wrapper
�
trace_0
trace_12�
+__inference_polymorphic_action_fn_146589301
+__inference_polymorphic_action_fn_146589370�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_02�
1__inference_polymorphic_distribution_fn_146589414�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
'__inference_get_initial_state_146589417�
���
FullArgSpec
args�
j
batch_size
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
 ztrace_0
�B�
__inference_<lambda>_146588473"�
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
__inference_<lambda>_146588471"�
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
`

action
get_initial_state
get_train_step
get_metadata"
signature_map
9:7Qd2'QNetwork/EncodingNetwork/dense_8/kernel
3:1d2%QNetwork/EncodingNetwork/dense_8/bias
9:7dP2'QNetwork/EncodingNetwork/dense_9/kernel
3:1P2%QNetwork/EncodingNetwork/dense_9/bias
::8P<2(QNetwork/EncodingNetwork/dense_10/kernel
4:2<2&QNetwork/EncodingNetwork/dense_10/bias
*:(<Q2QNetwork/dense_11/kernel
$:"Q2QNetwork/dense_11/bias
.

_q_network"
_generic_user_object
�B�
+__inference_polymorphic_action_fn_146589301	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_polymorphic_action_fn_146589370time_step_step_typetime_step_rewardtime_step_discounttime_step_observation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_polymorphic_distribution_fn_146589414	step_typerewarddiscountobservation"�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_get_initial_state_146589417
batch_size"�
���
FullArgSpec
args�
j
batch_size
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
�B�
?__inference_signature_wrapper_function_with_signature_146589204
0/discount0/observation0/reward0/step_type"�
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
�B�
?__inference_signature_wrapper_function_with_signature_146589214
batch_size"�
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
�B�
?__inference_signature_wrapper_function_with_signature_146589227"�
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
�B�
?__inference_signature_wrapper_function_with_signature_146589232"�
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
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_encoder
$_q_value_layer"
_tf_keras_layer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_postprocessing_layers"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecD
args<�9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
<
<0
=1
>2
?3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
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
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�2��
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
�2��
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
trackable_dict_wrapperF
__inference_<lambda>_146588471$�

� 
� "�
unknown 	6
__inference_<lambda>_146588473�

� 
� "� T
'__inference_get_initial_state_146589417)"�
�
�

batch_size 
� "� �
+__inference_polymorphic_action_fn_146589301����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������8
observation)�&
observation���������		
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
+__inference_polymorphic_action_fn_146589370����
���
���
TimeStep6
	step_type)�&
time_step_step_type���������0
reward&�#
time_step_reward���������4
discount(�%
time_step_discount���������B
observation3�0
time_step_observation���������		
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
1__inference_polymorphic_distribution_fn_146589414����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������8
observation)�&
observation���������		
� 
� "���

PolicyStep�
action������
`
B�?

atol� 

loc����������

rtol� 
L�I

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
�
j
parameters
� 
�
jname+tfp.distributions.Deterministic_ACTTypeSpec 
state� 
info� �
?__inference_signature_wrapper_function_with_signature_146589204����
� 
���
2
arg_0_discount �

0/discount���������
@
arg_0_observation+�(
0/observation���������		
.
arg_0_reward�
0/reward���������
4
arg_0_step_type!�
0/step_type���������"+�(
&
action�
action���������z
?__inference_signature_wrapper_function_with_signature_14658921470�-
� 
&�#
!

batch_size�

batch_size "� s
?__inference_signature_wrapper_function_with_signature_1465892270�

� 
� "�

int64�
int64 	W
?__inference_signature_wrapper_function_with_signature_146589232�

� 
� "� 