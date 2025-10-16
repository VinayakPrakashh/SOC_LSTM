module bram_burst #(
    parameter DATA_WIDTH = 12,
    parameter ADDR_WIDTH = 10,
    parameter NUM_PORTS = 16,
    parameter MATRIX_WIDTH = 16
    )(
    input clk,
    input rst,
    input start,
    input [DATA_WIDTH-1:0] data_in,
    input [ADDR_WIDTH-1:0] base_addr,
    input [ADDR_WIDTH-1:0] base_addr_row,
    input [ADDR_WIDTH-1:0] waddr_row,
    input [DATA_WIDTH-1:0] data_row_in,
    output reg [DATA_WIDTH-1:0] data_row_out,
    output reg [ADDR_WIDTH-1:0] row_addr,
    output reg wr_en_row,
    output reg we,
    output reg [DATA_WIDTH-1:0] data_out_0,
    output reg [DATA_WIDTH-1:0] data_out_1,
    output reg [DATA_WIDTH-1:0] data_out_2,
    output reg [DATA_WIDTH-1:0] data_out_3,
    output reg [DATA_WIDTH-1:0] data_out_4,
    output reg [DATA_WIDTH-1:0] data_out_5,
    output reg [DATA_WIDTH-1:0] data_out_6,
    output reg [DATA_WIDTH-1:0] data_out_7,
    output reg [DATA_WIDTH-1:0] data_out_8,
    output reg [DATA_WIDTH-1:0] data_out_9,
    output reg [DATA_WIDTH-1:0] data_out_10,
    output reg [DATA_WIDTH-1:0] data_out_11,
    output reg [DATA_WIDTH-1:0] data_out_12,
    output reg [DATA_WIDTH-1:0] data_out_13,
    output reg [DATA_WIDTH-1:0] data_out_14,
    output reg [DATA_WIDTH-1:0] data_out_15,
    output reg [ADDR_WIDTH-1:0] current_addr,
    output reg done,
    output reg [ADDR_WIDTH-1:0] waddr
);

parameter IDLE = 3'b000, READ = 3'b001, WRITE = 3'b010, NEXT_ROW = 3'b011, DONE = 3'b100;
reg [2:0] state, next_state;
reg [DATA_WIDTH-1:0] ram [0:NUM_PORTS-1];
reg [4-1:0] count_main, row_cnt;
// FIXED: Track current row base address
reg [ADDR_WIDTH-1:0] row_base_addr;

always @(posedge clk) begin
    if(rst) begin
        state <= IDLE;
    end
    else state <= next_state;
end

always @(*) begin
    case(state)
        IDLE: begin
            if (start) begin
                next_state = READ;
            end else begin
                next_state = IDLE;
            end
        end
        READ: begin
            if (count_main == NUM_PORTS-1) begin
                next_state = WRITE;
            end else begin
                next_state = READ;
            end
        end
        WRITE: begin
            next_state = NEXT_ROW;
        end
        NEXT_ROW: begin
            if (row_cnt == NUM_PORTS-1) begin
                next_state = DONE;  // All 16 rows done
            end else begin
                next_state = READ;  // Process next row
            end
        end
        DONE: begin
            next_state = IDLE;
        end
        default: next_state = IDLE;
    endcase
end

always @(posedge clk) begin
    case(state)
        IDLE: begin
            wr_en_row <= 0;
            row_base_addr <= base_addr; // Initialize row base address
            count_main <= 0;
            current_addr <= base_addr;
            row_addr <= base_addr_row;  // FIXED: Track row start
            done <= 0;
            we <= 0;
            row_cnt <= 0;
            waddr <= 0;
            data_out_0 <= 0;
            data_out_1 <= 0;
            data_out_2 <= 0;
            data_out_3 <= 0;
            data_out_4 <= 0;
            data_out_5 <= 0;
            data_out_6 <= 0;
            data_out_7 <= 0;
            data_out_8 <= 0;
            data_out_9 <= 0;
            data_out_10 <= 0;
            data_out_11 <= 0;
            data_out_12 <= 0;
            data_out_13 <= 0;
            data_out_14 <= 0;
            data_out_15 <= 0;
        end
        READ: begin
            ram[count_main] <= data_in;

            // FIXED: Increment address for next column in same row
            if (count_main < NUM_PORTS - 1) begin
                current_addr <= current_addr + 1;  // Move to next column in same row
                count_main <= count_main + 1;
            end else begin
                count_main <= 0;  // Reset counter after reading row
            end
        end
        WRITE: begin
            we <= 1;
            waddr <= row_cnt;
            wr_en_row <= 1;
            data_row_out <= data_row_in;
            data_out_0 <= ram[0];
            data_out_1 <= ram[1];
            data_out_2 <= ram[2];
            data_out_3 <= ram[3];
            data_out_4 <= ram[4];
            data_out_5 <= ram[5];
            data_out_6 <= ram[6];
            data_out_7 <= ram[7];
            data_out_8 <= ram[8];
            data_out_9 <= ram[9];
            data_out_10 <= ram[10];
            data_out_11 <= ram[11];
            data_out_12 <= ram[12];
            data_out_13 <= ram[13];
            data_out_14 <= ram[14];
            data_out_15 <= ram[15];
        end
        NEXT_ROW: begin
            we <= 0;
            wr_en_row <= 0;
            row_addr <= row_addr + 1;
            if (row_cnt < NUM_PORTS - 1) begin
                row_cnt <= row_cnt + 1;
                // FIXED: Move to next row (add MATRIX_WIDTH to get to same column of next row)
                row_base_addr <= row_base_addr + MATRIX_WIDTH;
                current_addr <= row_base_addr + MATRIX_WIDTH;  // Start of next row
            end else begin
                row_cnt <= 0; // Reset for next full operation
            end
        end
        DONE: begin
            done <= 1;
            we <= 0;
            wr_en_row <= 0;
        end
    endcase
end

endmodule