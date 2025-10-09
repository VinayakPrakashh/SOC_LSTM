module tiling_top #(
    parameter DATA_WIDTH = 12,
    parameter TILE_SIZE  = 16
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire [DATA_WIDTH-1:0]  data_in,
    output wire [DATA_WIDTH-1:0]  data_out
);
reg wr_en_1,wr_en_2;
reg [DATA_WIDTH-1:0] counter;
reg [1:0] state,next_state;
    tiled_bram_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_SIZE)
    ) bram_inst (
        .clk(clk),
        .rst(rst),
        .wr_en(wr_en_1),
        .data_in(data_in),
        .data_out(data_out)
    );

    tiled_bram_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_SIZE)
    ) bram_inst2 (
        .clk(clk),
        .rst(rst),   
        .wr_en(wr_en_2),
        .data_in(data_in),
        .data_out(data_out) // Not used in this example
    );
parameter IDLE=2'd0,INITIAL_LOAD=2'd1,OVERLAP1=2'd2,OVERLAP2=2'd3;
always @(*)begin
    case(state)
        IDLE:begin
            next_state=INITIAL_LOAD;
        end
        INITIAL_LOAD:begin
           if(counter==15)
             next_state=OVERLAP1;
           else
            next_state=INITIAL_LOAD;
        end
        OVERLAP1:begin
           if(counter==15)
             next_state=OVERLAP2;
           else
            next_state=OVERLAP1;
        end
        OVERLAP2:begin
            if(counter==15)
             next_state=OVERLAP1;
           else
            next_state=OVERLAP2;
        end
        default:next_state=IDLE;
    endcase
end 
always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        counter <= 0;
    if(counter==15)
        counter <=0;
    end else begin
        state <= next_state;
        counter <=counter+1;
    end 
end
always @(posedge clk or posedge rst) begin
    if (rst) begin
        wr_en_1 <=1'b0;
        wr_en_2 <=1'b0;
    end else begin
        case(next_state)
            IDLE:begin
                wr_en_1 <=1'b0;
                wr_en_2 <=1'b0;
            end
            INITIAL_LOAD:begin
                wr_en_1 <=1'b1;
                wr_en_2 <=1'b0;
            end
            OVERLAP2:begin
                wr_en_1 <=1'b1;
                wr_en_2 <=1'b0;
               
            end
            OVERLAP1:begin
              wr_en_1 <=1'b0;
              wr_en_2 <=1'b1;
            end
            default:begin
                wr_en_1 <=1'b0;
                wr_en_2 <=1'b0;
            end
        endcase
    end
end
endmodule