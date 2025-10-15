module sync_fifo #(
    parameter DATA_WIDTH = 12,
    parameter DEPTH      = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire wr_en,
    input  wire rd_en,                      
    input  wire [DATA_WIDTH-1:0] data_in,
    output reg  [DATA_WIDTH-1:0] data_out,
    output wire full,
    output wire empty
);

    localparam ADDR_WIDTH = $clog2(DEPTH);

    // FIFO memory
    reg [DATA_WIDTH-1:0] fifo_mem [0:DEPTH-1];

    // Pointers (extra MSB bit for wrap detection)
    reg [ADDR_WIDTH:0] wr_ptr, rd_ptr;

    // Flags
    assign full  = (wr_ptr[ADDR_WIDTH]     != rd_ptr[ADDR_WIDTH]) &&
                   (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);

    assign empty = (wr_ptr == rd_ptr);

    // Sequential logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_ptr   <= 0;
            rd_ptr   <= 0;
            data_out <= {DATA_WIDTH{1'b0}};
        end 
        else begin
            // Write operation
            if (wr_en && !full) begin
                fifo_mem[wr_ptr[ADDR_WIDTH-1:0]] <= data_in;
                wr_ptr <= wr_ptr + 1'b1;
            end

            // Read operation - FIXED: Added the missing logic
            if (rd_en && !empty) begin
                data_out <= fifo_mem[rd_ptr[ADDR_WIDTH-1:0]];
                rd_ptr   <= rd_ptr + 1'b1;
            end
        end
    end
endmodule