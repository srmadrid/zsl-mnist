const std = @import("std");

const zsl = @import("zsl");

const MLP = @This();

w1: zsl.array.Dense(f32),
b1: zsl.array.Dense(f32),
w2: zsl.array.Dense(f32),
b2: zsl.array.Dense(f32),

w1_var: zsl.array.Dense(zsl.autodiff.Var(f32)),
b1_var: zsl.array.Dense(zsl.autodiff.Var(f32)),
w2_var: zsl.array.Dense(zsl.autodiff.Var(f32)),
b2_var: zsl.array.Dense(zsl.autodiff.Var(f32)),

pub fn init(gpa: std.mem.Allocator, prng: std.Random, input: usize, hidden: usize, output: usize) !MLP {
    const normal: zsl.stats.Normal(f32) = .init(0.0, 0.01);

    var w1: zsl.array.Dense(f32) = try .initFn(gpa, &.{ input, hidden }, zsl.stats.Normal(f32).sample, .{ normal, prng }, .c);
    errdefer w1.deinit(gpa);
    var b1: zsl.array.Dense(f32) = try .initFn(gpa, &.{hidden}, zsl.stats.Normal(f32).sample, .{ normal, prng }, .c);
    errdefer b1.deinit(gpa);
    var w2: zsl.array.Dense(f32) = try .initFn(gpa, &.{ hidden, output }, zsl.stats.Normal(f32).sample, .{ normal, prng }, .c);
    errdefer w2.deinit(gpa);
    var b2: zsl.array.Dense(f32) = try .initFn(gpa, &.{output}, zsl.stats.Normal(f32).sample, .{ normal, prng }, .c);
    errdefer b2.deinit(gpa);

    var w1_var: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .init(gpa, &.{ input, hidden }, .c);
    errdefer w1_var.deinit(gpa);
    var b1_var: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .init(gpa, &.{hidden}, .c);
    errdefer b1_var.deinit(gpa);
    var w2_var: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .init(gpa, &.{ hidden, output }, .c);
    errdefer w2_var.deinit(gpa);
    var b2_var: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .init(gpa, &.{output}, .c);
    errdefer b2_var.deinit(gpa);

    return .{
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .w1_var = w1_var,
        .b1_var = b1_var,
        .w2_var = w2_var,
        .b2_var = b2_var,
    };
}

pub fn deinit(self: *MLP, gpa: std.mem.Allocator) void {
    self.w1.deinit(gpa);
    self.b1.deinit(gpa);
    self.w2.deinit(gpa);
    self.b2.deinit(gpa);

    self.w1_var.deinit(gpa);
    self.b1_var.deinit(gpa);
    self.w2_var.deinit(gpa);
    self.b2_var.deinit(gpa);

    self.* = undefined;
}

pub fn forward(self: *MLP, gpa: std.mem.Allocator, tape: *zsl.autodiff.Tape(f32), input: zsl.array.Dense(f32)) !zsl.array.Dense(zsl.autodiff.Var(f32)) {
    const VarCtx = struct {
        tape: *zsl.autodiff.Tape(f32),

        pub const is_numeric = true;

        fn bind(o: *zsl.autodiff.Var(f32), this: @This(), val: f32) void {
            o.* = .init(this.tape, val);
        }
    };

    const var_ctx: VarCtx = .{ .tape = tape };

    zsl.array.apply2IntoUnchecked(&self.w1_var, var_ctx, self.w1, VarCtx.bind);
    zsl.array.apply2IntoUnchecked(&self.b1_var, var_ctx, self.b1, VarCtx.bind);
    zsl.array.apply2IntoUnchecked(&self.w2_var, var_ctx, self.w2, VarCtx.bind);
    zsl.array.apply2IntoUnchecked(&self.b2_var, var_ctx, self.b2, VarCtx.bind);

    var z: zsl.vector.Dense(zsl.autodiff.Var(f32)) = try zsl.linalg.matmulAlloc(
        gpa,
        (input.ravelView(.c) catch unreachable).vectorView() catch unreachable,
        self.w1_var.matrixView(.row_major) catch unreachable,
    );
    defer z.deinit(gpa);

    zsl.vector.addIntoUnchecked(&z, z, self.b1_var.vectorView() catch unreachable);

    var z_arr = z.arrayView();
    zsl.array.apply1IntoUnchecked(&z_arr, z_arr, relu);

    var pred: zsl.vector.Dense(zsl.autodiff.Var(f32)) = try zsl.linalg.matmulAlloc(
        gpa,
        z,
        self.w2_var.matrixView(.row_major) catch unreachable,
    );
    errdefer pred.deinit(gpa);

    zsl.vector.addIntoUnchecked(&pred, pred, self.b2_var.vectorView() catch unreachable);

    var pred_arr = pred.arrayView();
    pred_arr.flags.owns_data = true;

    return pred_arr;
}

pub fn infer(self: MLP, gpa: std.mem.Allocator, input: zsl.array.Dense(f32)) !zsl.array.Dense(f32) {
    var z: zsl.vector.Dense(f32) = try zsl.linalg.matmulAlloc(
        gpa,
        (input.ravelView(.c) catch unreachable).vectorView() catch unreachable,
        self.w1.matrixView(.row_major) catch unreachable,
    );
    defer z.deinit(gpa);

    zsl.vector.addIntoUnchecked(&z, z, self.b1.vectorView() catch unreachable);

    var z_arr = z.arrayView();
    zsl.array.apply1IntoUnchecked(&z_arr, z_arr, relu);

    var pred: zsl.vector.Dense(f32) = try zsl.linalg.matmulAlloc(
        gpa,
        z,
        self.w2.matrixView(.row_major) catch unreachable,
    );
    errdefer pred.deinit(gpa);

    zsl.vector.addIntoUnchecked(&pred, pred, self.b2.vectorView() catch unreachable);

    var pred_arr = pred.arrayView();
    pred_arr.flags.owns_data = true;

    return pred_arr;
}

pub fn step(self: *MLP, lr: f32) void {
    const applyGrad = struct {
        fn call(weight: *f32, node: zsl.autodiff.Var(f32), step_lr: f32) void {
            weight.* = weight.* - step_lr * node.getGrad();
        }
    }.call;

    zsl.array.apply2IntoUnchecked(&self.w1, self.w1_var, lr, applyGrad);
    zsl.array.apply2IntoUnchecked(&self.b1, self.b1_var, lr, applyGrad);
    zsl.array.apply2IntoUnchecked(&self.w2, self.w2_var, lr, applyGrad);
    zsl.array.apply2IntoUnchecked(&self.b2, self.b2_var, lr, applyGrad);
}

fn relu(o: anytype, x: anytype) void {
    zsl.numeric.maxInto(o, 0, x);
}
