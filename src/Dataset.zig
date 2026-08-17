const std = @import("std");

const zsl = @import("zsl");

const Image = @import("Image.zig");

const Dataset = @This();

images: []Image,

pub fn init(arena: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, directory: []const u8) !Dataset {
    const labels_path = try std.fs.path.join(gpa, &.{ directory, "labels.txt" });
    defer gpa.free(labels_path);
    const labels_contents = try std.Io.Dir.cwd().readFileAlloc(io, labels_path, gpa, .unlimited);
    defer gpa.free(labels_contents);

    var labels: std.ArrayList(usize) = .empty;
    defer labels.deinit(gpa);

    const labels_trimmed = std.mem.trim(u8, labels_contents, " \t\r\n");
    var labels_iter = std.mem.splitScalar(u8, labels_trimmed, ',');
    while (labels_iter.next()) |chunk| {
        const chunk_trimmed = std.mem.trim(u8, chunk, " \t\r\n");

        if (chunk_trimmed.len == 0)
            continue;

        const label = try std.fmt.parseInt(usize, chunk_trimmed, 10);
        try labels.append(gpa, label);
    }

    const dataset: Dataset = .{ .images = try gpa.alloc(Image, labels.items.len) };
    errdefer gpa.free(dataset.images);
    const loaded = try gpa.alloc(bool, labels.items.len);
    defer gpa.free(loaded);
    @memset(loaded, false);

    const dir = try std.Io.Dir.cwd().openDir(io, directory, .{ .iterate = true });
    defer dir.close(io);

    var dir_iterator = dir.iterate();
    while (try dir_iterator.next(io)) |file| {
        if (!std.mem.endsWith(u8, file.name, ".pgm"))
            continue;

        const file_name = file.name[0 .. file.name.len - ".pgm".len];
        const idx = std.fmt.parseInt(usize, file_name, 10) catch continue;
        if (idx >= labels.items.len)
            continue;

        const file_contents = try dir.readFileAlloc(io, file.name, gpa, .unlimited);
        defer gpa.free(file_contents);

        dataset.images[idx] = try .init(arena, file_contents, labels.items[idx]);
        loaded[idx] = true;
    }

    for (loaded) |l| {
        if (!l)
            return error.MissingImageFile;
    }

    return dataset;
}

pub fn deinit(self: *Dataset, arena: std.mem.Allocator, gpa: std.mem.Allocator) void {
    for (self.images) |*image| {
        image.deinit(arena);
    }

    gpa.free(self.images);

    self.* = undefined;
}
