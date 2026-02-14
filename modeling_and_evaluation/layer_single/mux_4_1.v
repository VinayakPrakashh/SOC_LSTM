// 4:1 Multiplexer - If-Else Style
module mux_4to1 #(
    parameter WIDTH = 16
) (
    input  [WIDTH-1:0] in0,
    input  [WIDTH-1:0] in1,
    input  [WIDTH-1:0] in2,
    input  [WIDTH-1:0] in3,
    input  [1:0] sel,
    output reg [WIDTH-1:0] out
);

always @(*) begin
    if (sel == 2'b00)
        out = in0;
    else if (sel == 2'b01)
        out = in1;
    else if (sel == 2'b10)
        out = in2;
    else
        out = in3;
end

endmodule