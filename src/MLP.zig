const std = @import("std");

const zsl = @import("zsl");

const MLP = @This();

w1: zsl.array.Dense(zsl.autodiff.Var(f32)),
b1: zsl.array.Dense(zsl.autodiff.Var(f32)),
w2: zsl.array.Dense(zsl.autodiff.Var(f32)),
b2: zsl.array.Dense(zsl.autodiff.Var(f32)),

pub fn init(gpa: std.mem.Allocator, prng: std.Random, tape: *zsl.autodiff.Tape(f32), input: usize, hidden: usize, output: usize) !MLP {
    const Sampler = struct {
        tape: *zsl.autodiff.Tape(f32),
        prng: std.Random,
        normal: zsl.stats.Normal(f32) = .init(0.0, 0.01),

        fn call(self: @This()) zsl.autodiff.Var(f32) {
            const val: zsl.autodiff.Var(f32) = .init(self.tape, self.normal.sample(self.prng));
            self.tape.clear();
            return val;
        }
    };

    const sampler: Sampler = .{ .tape = tape, .prng = prng };

    var w1: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .initFn(gpa, &.{ input, hidden }, Sampler.call, .{sampler}, .c);
    errdefer w1.deinit(gpa);
    var b1: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .initFn(gpa, &.{hidden}, Sampler.call, .{sampler}, .c);
    errdefer b1.deinit(gpa);
    var w2: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .initFn(gpa, &.{ hidden, output }, Sampler.call, .{sampler}, .c);
    errdefer w2.deinit(gpa);
    var b2: zsl.array.Dense(zsl.autodiff.Var(f32)) = try .initFn(gpa, &.{output}, Sampler.call, .{sampler}, .c);
    errdefer b2.deinit(gpa);

    return .{
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
    };
}

pub fn deinit(self: *MLP, gpa: std.mem.Allocator) void {
    self.w1.deinit(gpa);
    self.b1.deinit(gpa);
    self.w2.deinit(gpa);
    self.b2.deinit(gpa);

    self.* = undefined;
}

pub fn zeroGrads(self: *MLP) void {
    const zeroGrad = struct {
        fn call(o: *zsl.autodiff.Var(f32), x: zsl.autodiff.Var(f32)) void {
            o.* = x;
            o.setGrad(0.0);
        }
    }.call;

    zsl.array.apply1Into(&self.w1, self.w1, zeroGrad) catch unreachable;
    zsl.array.apply1Into(&self.b1, self.b1, zeroGrad) catch unreachable;
    zsl.array.apply1Into(&self.w2, self.w2, zeroGrad) catch unreachable;
    zsl.array.apply1Into(&self.b2, self.b2, zeroGrad) catch unreachable;
}

pub fn forward(self: MLP, gpa: std.mem.Allocator, input: zsl.array.Dense(f32)) !zsl.array.Dense(zsl.autodiff.Var(f32)) {
    var z: zsl.vector.Dense(zsl.autodiff.Var(f32)) = try zsl.linalg.matmulAlloc(
        gpa,
        (input.ravelView(.c) catch unreachable).vectorView() catch unreachable,
        self.w1.matrixView(.row_major) catch unreachable,
    );
    defer z.deinit(gpa);

    try zsl.vector.addInto(&z, z, self.b1.vectorView() catch unreachable);

    var z_arr = z.arrayView();
    try zsl.array.apply1Into(&z_arr, z_arr, relu);

    var pred: zsl.vector.Dense(zsl.autodiff.Var(f32)) = try zsl.linalg.matmulAlloc(
        gpa,
        z,
        self.w2.matrixView(.row_major) catch unreachable,
    );
    errdefer pred.deinit(gpa);

    try zsl.vector.addInto(&pred, pred, self.b2.vectorView() catch unreachable);

    var pred_arr = pred.arrayView();
    pred_arr.flags.owns_data = true;

    return pred_arr;
}

pub fn infer(self: MLP, gpa: std.mem.Allocator, input: zsl.array.Dense(f32)) !zsl.array.Dense(f32) {
    var w1_f: zsl.array.Dense(f32) = try zsl.array.apply1Alloc(gpa, self.w1, zsl.autodiff.Var(f32).getVal);
    defer w1_f.deinit(gpa);
    var b1_f: zsl.array.Dense(f32) = try zsl.array.apply1Alloc(gpa, self.b1, zsl.autodiff.Var(f32).getVal);
    defer b1_f.deinit(gpa);
    var w2_f: zsl.array.Dense(f32) = try zsl.array.apply1Alloc(gpa, self.w2, zsl.autodiff.Var(f32).getVal);
    defer w2_f.deinit(gpa);
    var b2_f: zsl.array.Dense(f32) = try zsl.array.apply1Alloc(gpa, self.b2, zsl.autodiff.Var(f32).getVal);
    defer b2_f.deinit(gpa);

    var z: zsl.vector.Dense(f32) = try zsl.linalg.matmulAlloc(
        gpa,
        (input.ravelView(.c) catch unreachable).vectorView() catch unreachable,
        w1_f.matrixView(.row_major) catch unreachable,
    );
    defer z.deinit(gpa);

    try zsl.vector.addInto(&z, z, b1_f.vectorView() catch unreachable);

    var z_arr = z.arrayView();
    try zsl.array.apply1Into(&z_arr, z_arr, relu);

    var pred: zsl.vector.Dense(f32) = try zsl.linalg.matmulAlloc(
        gpa,
        z,
        w2_f.matrixView(.row_major) catch unreachable,
    );
    errdefer pred.deinit(gpa);

    try zsl.vector.addInto(&pred, pred, b2_f.vectorView() catch unreachable);

    var pred_arr = pred.arrayView();
    pred_arr.flags.owns_data = true;

    return pred_arr;
}

pub fn step(self: *MLP, lr: f32) void {
    const applyGrad = struct {
        fn call(o: *zsl.autodiff.Var(f32), x: zsl.autodiff.Var(f32), y: f32) void {
            o.* = x;
            o.setVal(o.getVal() - y * o.getGrad());
        }
    }.call;

    zsl.array.apply2Into(&self.w1, self.w1, lr, applyGrad) catch unreachable;
    zsl.array.apply2Into(&self.b1, self.b1, lr, applyGrad) catch unreachable;
    zsl.array.apply2Into(&self.w2, self.w2, lr, applyGrad) catch unreachable;
    zsl.array.apply2Into(&self.b2, self.b2, lr, applyGrad) catch unreachable;
}

fn relu(o: anytype, x: anytype) void {
    zsl.numeric.maxInto(o, 0, x);
}
