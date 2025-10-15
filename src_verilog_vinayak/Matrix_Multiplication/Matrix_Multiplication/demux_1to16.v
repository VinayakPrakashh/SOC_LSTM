module demux_1to16#(
    parameter DATA_WIDTH = 12
)(
    input wire [DATA_WIDTH-1:0] data_in,
    input wire [4:0] select,  // 5 bits to select one of 32 outputs
    output wire [DATA_WIDTH-1:0] data_out_0,
    output wire [DATA_WIDTH-1:0] data_out_1,
    output wire [DATA_WIDTH-1:0] data_out_2,
    output wire [DATA_WIDTH-1:0] data_out_3,
    output wire [DATA_WIDTH-1:0] data_out_4,
    output wire [DATA_WIDTH-1:0] data_out_5,
    output wire [DATA_WIDTH-1:0] data_out_6,
    output wire [DATA_WIDTH-1:0] data_out_7,
    output wire [DATA_WIDTH-1:0] data_out_8,
    output wire [DATA_WIDTH-1:0] data_out_9,
    output wire [DATA_WIDTH-1:0] data_out_10,
    output wire [DATA_WIDTH-1:0] data_out_11,
    output wire [DATA_WIDTH-1:0] data_out_12,
    output wire [DATA_WIDTH-1:0] data_out_13,
    output wire [DATA_WIDTH-1:0] data_out_14,
    output wire [DATA_WIDTH-1:0] data_out_15,
);

    assign data_out_0  = (select == 5'd0)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_1  = (select == 5'd1)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_2  = (select == 5'd2)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_3  = (select == 5'd3)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_4  = (select == 5'd4)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_5  = (select == 5'd5)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_6  = (select == 5'd6)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_7  = (select == 5'd7)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_8  = (select == 5'd8)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_9  = (select == 5'd9)  ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_10 = (select == 5'd10) ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_11 = (select == 5'd11) ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_12 = (select == 5'd12) ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_13 = (select == 5'd13) ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_14 = (select == 5'd14) ? data_in : {DATA_WIDTH{1'b0}};
    assign data_out_15 = (select == 5'd15) ? data_in : {DATA_WIDTH{1'b0}};


endmodule