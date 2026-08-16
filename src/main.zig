const std = @import("std");

const Image = @import("Image.zig");
const Dataset = @import("Dataset.zig");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;

    var xoshiro: std.Random.Xoshiro256 = .init(@bitCast(std.Io.Clock.real.now(io).toMicroseconds()));
    const prng = xoshiro.random();

    var train_data: Dataset = try .init(arena, gpa, io, "data/train");
    defer train_data.deinit(arena, gpa);
    var test_data: Dataset = try .init(arena, gpa, io, "data/test");
    defer test_data.deinit(arena, gpa);

    std.debug.print("Random number: {f}\n", .{train_data.getAssumeInBounds(prng.intRangeAtMost(usize, 0, test_data.images.len - 1))});
}
