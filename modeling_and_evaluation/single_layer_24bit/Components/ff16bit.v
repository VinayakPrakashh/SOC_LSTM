module dff_16bit (
    input wire clk,
    input wire [23:0] d,
    output reg [23:0] q
);

always @(posedge clk) begin
    q <= d;
end

endmodule