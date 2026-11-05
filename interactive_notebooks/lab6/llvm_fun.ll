; Function Signature: polynomial(Array{Float64, 1}, Float64)
define double @julia_polynomial_10831(ptr noundef nonnull align 8 dereferenceable(24) %"a::Array", double %"x::Float64") local_unnamed_addr #0 {
top:
  %"new::Tuple" = alloca [1 x i64], align 8
  %"new::Tuple34" = alloca [1 x i64], align 8
  %"a::Array.size_ptr" = getelementptr inbounds nuw i8, ptr %"a::Array", i64 16
  %"a::Array.size.0.copyload" = load i64, ptr %"a::Array.size_ptr", align 8
  %.not = icmp eq i64 %"a::Array.size.0.copyload", 0
  br i1 %.not, label %L13, label %L17

L13:                                              ; preds = %top
  store i64 0, ptr %"new::Tuple34", align 8
  call void @j_throw_boundserror_10834(ptr nonnull %"a::Array", ptr nocapture nonnull readonly %"new::Tuple34") #10
  unreachable

L17:                                              ; preds = %top
  %0 = add i64 %"a::Array.size.0.copyload", -1
  %memoryref_data = load ptr, ptr %"a::Array", align 8
  %memoryref_byteoffset = shl i64 %0, 3
  %memoryref_data4 = getelementptr inbounds i8, ptr %memoryref_data, i64 %memoryref_byteoffset
  %1 = load double, ptr %memoryref_data4, align 8
  %.not39 = icmp eq i64 %0, 1
  %.inv = icmp slt i64 %0, 1
  %spec.select = select i1 %.inv, i64 %"a::Array.size.0.copyload", i64 1
  %value_phi = select i1 %.not39, i64 1, i64 %spec.select
  %.not40.not = icmp slt i64 %0, %value_phi
  br i1 %.not40.not, label %L110, label %L76

L76:                                              ; preds = %L94, %L17
  %value_phi10 = phi i64 [ %2, %L94 ], [ %0, %L17 ]
  %value_phi12 = phi double [ %5, %L94 ], [ %1, %L17 ]
  %2 = add i64 %value_phi10, -1
  %.not41 = icmp ult i64 %2, %"a::Array.size.0.copyload"
  br i1 %.not41, label %L94, label %L90

L90:                                              ; preds = %L76
  store i64 %value_phi10, ptr %"new::Tuple", align 8
  call void @j_throw_boundserror_10834(ptr nonnull %"a::Array", ptr nocapture nonnull readonly %"new::Tuple") #10
  unreachable

L94:                                              ; preds = %L76
  %3 = fmul double %"x::Float64", %value_phi12
  %memoryref_byteoffset20 = shl i64 %2, 3
  %memoryref_data25 = getelementptr inbounds i8, ptr %memoryref_data, i64 %memoryref_byteoffset20
  %4 = load double, ptr %memoryref_data25, align 8
  %5 = fadd double %3, %4
  %.not42 = icmp eq i64 %value_phi10, %value_phi
  br i1 %.not42, label %L110, label %L76

L110:                                             ; preds = %L94, %L17
  %value_phi29 = phi double [ %1, %L17 ], [ %5, %L94 ]
  ret double %value_phi29
}
