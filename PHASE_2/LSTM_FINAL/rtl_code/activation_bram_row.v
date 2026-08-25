module activation_bram_row #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16,
    parameter MEM_SIZE   = 94       // ✅ CHANGE: was DATA_MEM_SIZE, now GATE_MEM_SIZE
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   inter_rst,
    input  wire                   reset_done,
    input  wire                   we,
    input  wire [ADDR_WIDTH-1:0]  addr,
    input  wire [DATA_WIDTH-1:0]  din,
    input  wire                   rd_en,
    input  wire [ADDR_WIDTH-1:0]  rd_addr,
    output reg  [DATA_WIDTH-1:0]  dout,
    output wire                   done,
    output wire                   full,          // ✅ ADD: new full signal
    output wire                   read_done_out,
    output wire [1:0]             write_count
);

// ...existing code...
  // ----------------------------------------------------
    reg [DATA_WIDTH-1:0] bram [0:MEM_SIZE-1];
    reg [ADDR_WIDTH-1:0] read_count;    
    // Counter to track number of writes
    reg                done_write;
    reg read_done;
    assign done = (reset_done == 1'b0) ? done_write : 1'b0;
    reg [DATA_WIDTH-1:0] mem [0:MEM_SIZE-1];

    // ✅ ADD: write counter to track how many values written
    reg [6:0] wr_count;         // 7 bits enough for 94
    reg       bram_full;

    assign full  = bram_full;
    assign done  = bram_full;   // done = full (same signal)

    // ✅ CHANGE: Write logic with full detection
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || inter_rst) begin
            wr_count  <= 0;
            bram_full <= 0;
        end else begin

            // ✅ Reset full when reset_done pulses (after read is done)
            if (reset_done)
                bram_full <= 0;

            if (we && !bram_full) begin
                mem[addr] <= din;
                if (wr_count == MEM_SIZE - 1) begin
                    wr_count  <= 0;
                    bram_full <= 1;  // ✅ Assert full after MEM_SIZE writes
                end else begin
                    wr_count  <= wr_count + 1;
                end
            end
        end
    end

    // ...existing read logic unchanged...
    reg [6:0] rd_count;
    reg       bram_read_done;

    assign read_done_out = bram_read_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || inter_rst) begin
            rd_count      <= 0;
            bram_read_done<= 0;
            dout          <= 0;
        end else begin
            if (reset_done)
                bram_read_done <= 0;

            if (rd_en && bram_full) begin
                dout <= mem[rd_addr];
                if (rd_count == MEM_SIZE - 1) begin
                    rd_count       <= 0;
                    bram_read_done <= 1;  // ✅ read done after MEM_SIZE reads
                end else begin
                    rd_count <= rd_count + 1;
                end
            end
        end
    end

// ...existing code...

endmodule