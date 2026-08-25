module sync_fifo #(
    parameter DEPTH = 10,
    parameter WIDTH = 8
)(
    input  wire             clk,
    input  wire             rst_n,      // Active low reset
    input  wire             wr_en,      // Write enable
    input  wire             rd_en,      // Read enable
    input  wire [WIDTH-1:0] wr_data,    // Write data
    output reg  [WIDTH-1:0] rd_data,    // Read data
    output wire             full,       // FIFO full flag
    output wire             empty,      // FIFO empty flag
    output reg  [$clog2(DEPTH):0] count // Number of elements in FIFO
);

    // Memory array
    reg [WIDTH-1:0] fifo_mem [0:DEPTH-1];

    // Pointers
    reg [$clog2(DEPTH)-1:0] wr_ptr;
    reg [$clog2(DEPTH)-1:0] rd_ptr;

    // Full and Empty flags
    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    // Write Logic
    always @(posedge clk or posedge rst_n) begin
        if (rst_n) begin
            wr_ptr <= 0;
        end else if (wr_en && !full) begin
            fifo_mem[wr_ptr] <= wr_data;
            wr_ptr <= (wr_ptr == DEPTH-1) ? 0 : wr_ptr + 1;
        end
    end

    // Read Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr  <= 0;
            rd_data <= 0;
        end else if (rd_en && !empty) begin
            rd_data <= fifo_mem[rd_ptr];
            rd_ptr  <= (rd_ptr == DEPTH-1) ? 0 : rd_ptr + 1;
        end
    end

    // Count Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 0;
        end else begin
            case ({wr_en && !full, rd_en && !empty})
                2'b10:   count <= count + 1; // Write only
                2'b01:   count <= count - 1; // Read only
                default: count <= count;     // Both or neither
            endcase
        end
    end

endmodule