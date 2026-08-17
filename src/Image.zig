const std = @import("std");

const zsl = @import("zsl");

const Image = @This();

data: zsl.array.Dense(f32),
label: usize,

/// Initializes an `Image` from a given file. Assumes the filename is the index
/// of the label in a `labels.txt` file within the same directory.
pub fn init(arena: std.mem.Allocator, file_contents: []const u8, label: usize) !Image {
    if (file_contents.len < 2 or !std.mem.startsWith(u8, file_contents, "P5"))
        return error.InavilPgmMagicNumber;

    const skipWhitespaceAndComments = struct {
        fn call(buf: []const u8, pos: *usize) void {
            while (pos.* < buf.len) {
                const char = buf[pos.*];
                if (char == '#') {
                    while (pos.* < buf.len and buf[pos.*] != '\n') : (pos.* += 1) {}
                } else if (std.ascii.isWhitespace(char))
                    pos.* += 1
                else
                    break;
            }
        }
    }.call;

    const readInt = struct {
        fn call(buf: []const u8, pos: *usize) !usize {
            const start = pos.*;
            while (pos.* < buf.len and std.ascii.isDigit(buf[pos.*])) : (pos.* += 1) {}

            if (pos.* == start)
                return error.InvalidPgmHeader;

            return std.fmt.parseInt(usize, buf[start..pos.*], 10);
        }
    }.call;

    var pos: usize = 2;

    skipWhitespaceAndComments(file_contents, &pos);
    const width = try readInt(file_contents, &pos);
    skipWhitespaceAndComments(file_contents, &pos);
    const height = try readInt(file_contents, &pos);
    skipWhitespaceAndComments(file_contents, &pos);
    const maxval = try readInt(file_contents, &pos);

    if (pos >= file_contents.len or !std.ascii.isWhitespace(file_contents[pos]))
        return error.InvalidPgmHeader;

    pos += 1;

    if (maxval > 255)
        return error.UnsupportedPgmMaxval;

    const maxval_f = zsl.numeric.cast(f32, maxval);

    const pixel_count = width * height;
    if (file_contents.len < pos + pixel_count)
        return error.TruncatedPgmData;

    var image: Image = .{
        .data = try .init(arena, &.{ height, width }, .c),
        .label = label,
    };

    for (0..height) |i| {
        for (0..width) |j| {
            const byte = file_contents[pos + i * width + j];
            const val = zsl.numeric.cast(f32, byte) / maxval_f;
            image.data.setAssumeInBounds(&.{ i, j }, val);
        }
    }

    return image;
}

pub fn deinit(self: *Image, arena: std.mem.Allocator) void {
    self.data.deinit(arena);

    self.* = undefined;
}

pub fn format(self: Image, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("Label {d}\n", .{self.label});

    // const ramp = " .:-=+*#%@";
    const ramp = " .'^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

    for (0..self.data.rows) |i| {
        for (0..self.data.cols) |j| {
            const val = self.data.getAssumeInBounds(i, j);
            const clamped = std.math.clamp(val, 0.0, 1.0);
            const idx = zsl.numeric.cast(usize, clamped * zsl.numeric.cast(f32, ramp.len - 1));

            try writer.writeByte(ramp[idx]);
            try writer.writeByte(ramp[idx]);
        }

        try writer.writeByte('\n');
    }
}
