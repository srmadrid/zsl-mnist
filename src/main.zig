const std = @import("std");
const builtin = @import("builtin");

const zsl = @import("zsl");

const Image = @import("Image.zig");
const Dataset = @import("Dataset.zig");
const MLP = @import("MLP.zig");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;

    var xoshiro: std.Random.Xoshiro256 = .init(@bitCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const prng = xoshiro.random();

    // Load dataset
    var train_data: Dataset = try .init(arena, gpa, io, "data/train");
    defer train_data.deinit(arena, gpa);
    var test_data: Dataset = try .init(arena, gpa, io, "data/test");
    defer test_data.deinit(arena, gpa);

    const data_width = train_data.images[0].data.shape[1];
    const data_height = train_data.images[0].data.shape[0];
    var num_classes: usize = 0;

    for (train_data.images) |train_image| {
        if (train_image.label > num_classes)
            num_classes = train_image.label;
    }

    for (test_data.images) |test_image| {
        if (test_image.label > num_classes)
            num_classes = test_image.label;
    }

    // Initialize tape and perceptron
    var tape: zsl.autodiff.Tape(f32) = try .init(gpa, 500_000);
    defer tape.deinit(gpa);
    var model: MLP = try .init(gpa, prng, &tape, data_width * data_height, 128, num_classes);
    defer model.deinit(gpa);
    tape.clear();

    var pred = try model.forward(gpa, train_data.images[0].data);
    defer pred.deinit(gpa);
}
