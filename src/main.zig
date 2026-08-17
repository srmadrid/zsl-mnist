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

    var stdout_buffer: [2048]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    var stdout_writer = &(stdout.interface);

    // const seed: u64 = @bitCast(std.Io.Clock.real.now(io).toMicroseconds());
    var xoshiro: std.Random.Xoshiro256 = .init(42);
    const prng = xoshiro.random();

    var data_train: Dataset = try .init(arena, gpa, io, "data/train");
    defer data_train.deinit(arena, gpa);
    var data_test: Dataset = try .init(arena, gpa, io, "data/test");
    defer data_test.deinit(arena, gpa);

    const data_width = data_train.images[0].data.shape[1];
    const data_height = data_train.images[0].data.shape[0];
    var num_classes: usize = 0;

    for (data_train.images) |image| {
        if (image.label > num_classes)
            num_classes = image.label;
    }

    for (data_test.images) |image| {
        if (image.label > num_classes)
            num_classes = image.label;
    }

    num_classes += 1;

    var tape: zsl.autodiff.Tape(f32) = try .init(gpa, 500_000);
    defer tape.deinit(gpa);

    var model: MLP = try .init(gpa, prng, data_width * data_height, 128, num_classes);
    defer model.deinit(gpa);

    try stdout_writer.print(
        \\
        \\  =======================================================
        \\                    ZSL-MNIST TRAINING
        \\  =======================================================
        \\   EPOCH | LOSS       | TRAIN ACCURACY | TEST ACCURACY
        \\  -------------------------------------------------------
        \\
    , .{});
    try stdout_writer.flush();

    const lr: f32 = 0.01;
    const epochs = 5;
    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0.0;
        var correct_train: usize = 0;

        for (data_train.images, 0..) |image, idx| {
            if (idx % 100 == 0) {
                try stdout_writer.print(
                    "\r   {d:^5} | Training... [{d}/{d}]",
                    .{ epoch + 1, idx, data_train.images.len },
                );
                try stdout_writer.flush();
            }

            var pred = try model.forward(gpa, &tape, image.data);
            defer pred.deinit(gpa);

            var max_idx: usize = 0;
            var max_val: f32 = pred.getAssumeInBounds(&.{0}).getVal();
            for (1..num_classes) |i| {
                const val = pred.getAssumeInBounds(&.{i}).getVal();
                if (val > max_val) {
                    max_idx = i;
                    max_val = val;
                }
            }

            if (max_idx == image.label)
                correct_train += 1;

            var loss: zsl.autodiff.Var(f32) = .init(&tape, 0.0);
            for (0..num_classes) |i| {
                const val = pred.getAssumeInBounds(&.{i});
                const target: f32 = if (i == image.label) 1.0 else 0.0;

                const diff = zsl.numeric.sub(val, target);
                const sq_diff = zsl.numeric.mul(diff, diff);
                zsl.numeric.addInto(&loss, loss, sq_diff);
            }

            zsl.numeric.divInto(&loss, loss, num_classes);
            epoch_loss += loss.getVal();

            loss.backward();
            model.step(lr);

            tape.clear();
        }

        const avg_train_loss = zsl.numeric.div(epoch_loss, data_train.images.len);
        const acc_train = zsl.numeric.mul(zsl.numeric.div(zsl.numeric.cast(f32, correct_train), data_train.images.len), 100);

        var correct_test: usize = 0;
        for (data_test.images, 0..) |image, idx| {
            if (idx % 100 == 0) {
                try stdout_writer.print(
                    "\r   {d:^5} | Testing...  [{d}/{d}]",
                    .{ epoch + 1, idx, data_test.images.len },
                );
                try stdout_writer.flush();
            }

            var pred = try model.infer(gpa, image.data);
            defer pred.deinit(gpa);

            var predicted_class: usize = 0;
            var max_val: f32 = pred.getAssumeInBounds(&.{0});
            for (1..num_classes) |i| {
                const val = pred.getAssumeInBounds(&.{i});
                if (val > max_val) {
                    predicted_class = i;
                    max_val = val;
                }
            }

            if (predicted_class == image.label)
                correct_test += 1;
        }

        const acc_test = zsl.numeric.mul(zsl.numeric.div(zsl.numeric.cast(f32, correct_test), data_test.images.len), 100);

        try stdout_writer.print("\r   {d:^5} | {d:<10.4} | {d:>6.2}%        | {d:>6.2}%\n", .{
            epoch + 1,
            avg_train_loss,
            acc_train,
            acc_test,
        });
        try stdout_writer.flush();
    }

    try stdout_writer.print("  =======================================================\n\n", .{});
    try stdout_writer.flush();
}
