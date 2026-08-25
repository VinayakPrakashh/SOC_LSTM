module demux_1to10_16bit (
    input wire [15:0] data_in,
    input wire [3:0] sel,  // 4-bit select (0-9)
    
    output reg [15:0] out0,
    output reg [15:0] out1,
    output reg [15:0] out2,
    output reg [15:0] out3,
    output reg [15:0] out4,
    output reg [15:0] out5,
    output reg [15:0] out6,
    output reg [15:0] out7,
    output reg [15:0] out8,
    output reg [15:0] out9
);

    always @(*) begin
        // Default all outputs to 0
        out0 = 16'd0;
        out1 = 16'd0;
        out2 = 16'd0;
        out3 = 16'd0;
        out4 = 16'd0;
        out5 = 16'd0;
        out6 = 16'd0;
        out7 = 16'd0;
        out8 = 16'd0;
        out9 = 16'd0;
        
        case (sel)
            4'd0: out0 = data_in;
            4'd1: out1 = data_in;
            4'd2: out2 = data_in;
            4'd3: out3 = data_in;
            4'd4: out4 = data_in;
            4'd5: out5 = data_in;
            4'd6: out6 = data_in;
            4'd7: out7 = data_in;
            4'd8: out8 = data_in;
            4'd9: out9 = data_in;
            default: ; // All outputs remain 0
        endcase
    end

endmodule