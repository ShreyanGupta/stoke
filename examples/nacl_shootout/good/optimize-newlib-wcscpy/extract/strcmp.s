  .text
  .globl strcmp
  .type strcmp, @function

#! file-offset 0x166780
#! rip-offset  0x126780
#! capacity    256 bytes

# Text                         #  Line  RIP       Bytes  Opcode              
.strcmp:                       #        0x126780  0      OPC=<label>         
  movl %esi, %esi              #  1     0x126780  2      OPC=movl_r32_r32    
  movl %edi, %edi              #  2     0x126782  2      OPC=movl_r32_r32    
  movl %esi, %eax              #  3     0x126784  2      OPC=movl_r32_r32    
  orl %edi, %eax               #  4     0x126786  2      OPC=orl_r32_r32     
  testb $0x3, %al              #  5     0x126788  2      OPC=testb_al_imm8   
  je .L_126840                 #  6     0x12678a  6      OPC=je_label_1      
  nop                          #  7     0x126790  1      OPC=nop             
  nop                          #  8     0x126791  1      OPC=nop             
  nop                          #  9     0x126792  1      OPC=nop             
  nop                          #  10    0x126793  1      OPC=nop             
  nop                          #  11    0x126794  1      OPC=nop             
  nop                          #  12    0x126795  1      OPC=nop             
  nop                          #  13    0x126796  1      OPC=nop             
  nop                          #  14    0x126797  1      OPC=nop             
  nop                          #  15    0x126798  1      OPC=nop             
  nop                          #  16    0x126799  1      OPC=nop             
  nop                          #  17    0x12679a  1      OPC=nop             
  nop                          #  18    0x12679b  1      OPC=nop             
  nop                          #  19    0x12679c  1      OPC=nop             
  nop                          #  20    0x12679d  1      OPC=nop             
  nop                          #  21    0x12679e  1      OPC=nop             
  nop                          #  22    0x12679f  1      OPC=nop             
.L_1267a0:                     #        0x1267a0  0      OPC=<label>         
  movl %edi, %edi              #  23    0x1267a0  2      OPC=movl_r32_r32    
  movzbl (%r15,%rdi,1), %eax   #  24    0x1267a2  5      OPC=movzbl_r32_m8   
  testb %al, %al               #  25    0x1267a7  2      OPC=testb_r8_r8     
  jne .L_1267e0                #  26    0x1267a9  2      OPC=jne_label       
  jmpq .L_126800               #  27    0x1267ab  2      OPC=jmpq_label      
  nop                          #  28    0x1267ad  1      OPC=nop             
  nop                          #  29    0x1267ae  1      OPC=nop             
  nop                          #  30    0x1267af  1      OPC=nop             
  nop                          #  31    0x1267b0  1      OPC=nop             
  nop                          #  32    0x1267b1  1      OPC=nop             
  nop                          #  33    0x1267b2  1      OPC=nop             
  nop                          #  34    0x1267b3  1      OPC=nop             
  nop                          #  35    0x1267b4  1      OPC=nop             
  nop                          #  36    0x1267b5  1      OPC=nop             
  nop                          #  37    0x1267b6  1      OPC=nop             
  nop                          #  38    0x1267b7  1      OPC=nop             
  nop                          #  39    0x1267b8  1      OPC=nop             
  nop                          #  40    0x1267b9  1      OPC=nop             
  nop                          #  41    0x1267ba  1      OPC=nop             
  nop                          #  42    0x1267bb  1      OPC=nop             
  nop                          #  43    0x1267bc  1      OPC=nop             
  nop                          #  44    0x1267bd  1      OPC=nop             
  nop                          #  45    0x1267be  1      OPC=nop             
  nop                          #  46    0x1267bf  1      OPC=nop             
.L_1267c0:                     #        0x1267c0  0      OPC=<label>         
  addl $0x1, %edi              #  47    0x1267c0  3      OPC=addl_r32_imm8   
  addl $0x1, %esi              #  48    0x1267c3  3      OPC=addl_r32_imm8   
  movl %edi, %edi              #  49    0x1267c6  2      OPC=movl_r32_r32    
  movzbl (%r15,%rdi,1), %eax   #  50    0x1267c8  5      OPC=movzbl_r32_m8   
  testb %al, %al               #  51    0x1267cd  2      OPC=testb_r8_r8     
  je .L_126800                 #  52    0x1267cf  2      OPC=je_label        
  nop                          #  53    0x1267d1  1      OPC=nop             
  nop                          #  54    0x1267d2  1      OPC=nop             
  nop                          #  55    0x1267d3  1      OPC=nop             
  nop                          #  56    0x1267d4  1      OPC=nop             
  nop                          #  57    0x1267d5  1      OPC=nop             
  nop                          #  58    0x1267d6  1      OPC=nop             
  nop                          #  59    0x1267d7  1      OPC=nop             
  nop                          #  60    0x1267d8  1      OPC=nop             
  nop                          #  61    0x1267d9  1      OPC=nop             
  nop                          #  62    0x1267da  1      OPC=nop             
  nop                          #  63    0x1267db  1      OPC=nop             
  nop                          #  64    0x1267dc  1      OPC=nop             
  nop                          #  65    0x1267dd  1      OPC=nop             
  nop                          #  66    0x1267de  1      OPC=nop             
  nop                          #  67    0x1267df  1      OPC=nop             
.L_1267e0:                     #        0x1267e0  0      OPC=<label>         
  movl %esi, %esi              #  68    0x1267e0  2      OPC=movl_r32_r32    
  cmpb (%r15,%rsi,1), %al      #  69    0x1267e2  4      OPC=cmpb_r8_m8      
  je .L_1267c0                 #  70    0x1267e6  2      OPC=je_label        
  nop                          #  71    0x1267e8  1      OPC=nop             
  nop                          #  72    0x1267e9  1      OPC=nop             
  nop                          #  73    0x1267ea  1      OPC=nop             
  nop                          #  74    0x1267eb  1      OPC=nop             
  nop                          #  75    0x1267ec  1      OPC=nop             
  nop                          #  76    0x1267ed  1      OPC=nop             
  nop                          #  77    0x1267ee  1      OPC=nop             
  nop                          #  78    0x1267ef  1      OPC=nop             
  nop                          #  79    0x1267f0  1      OPC=nop             
  nop                          #  80    0x1267f1  1      OPC=nop             
  nop                          #  81    0x1267f2  1      OPC=nop             
  nop                          #  82    0x1267f3  1      OPC=nop             
  nop                          #  83    0x1267f4  1      OPC=nop             
  nop                          #  84    0x1267f5  1      OPC=nop             
  nop                          #  85    0x1267f6  1      OPC=nop             
  nop                          #  86    0x1267f7  1      OPC=nop             
  nop                          #  87    0x1267f8  1      OPC=nop             
  nop                          #  88    0x1267f9  1      OPC=nop             
  nop                          #  89    0x1267fa  1      OPC=nop             
  nop                          #  90    0x1267fb  1      OPC=nop             
  nop                          #  91    0x1267fc  1      OPC=nop             
  nop                          #  92    0x1267fd  1      OPC=nop             
  nop                          #  93    0x1267fe  1      OPC=nop             
  nop                          #  94    0x1267ff  1      OPC=nop             
.L_126800:                     #        0x126800  0      OPC=<label>         
  movl %edi, %edi              #  95    0x126800  2      OPC=movl_r32_r32    
  movzbl (%r15,%rdi,1), %eax   #  96    0x126802  5      OPC=movzbl_r32_m8   
  movl %esi, %esi              #  97    0x126807  2      OPC=movl_r32_r32    
  movzbl (%r15,%rsi,1), %edx   #  98    0x126809  5      OPC=movzbl_r32_m8   
  subl %edx, %eax              #  99    0x12680e  2      OPC=subl_r32_r32    
  popq %r11                    #  100   0x126810  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d      #  101   0x126812  7      OPC=andl_r32_imm32  
  nop                          #  102   0x126819  1      OPC=nop             
  nop                          #  103   0x12681a  1      OPC=nop             
  nop                          #  104   0x12681b  1      OPC=nop             
  nop                          #  105   0x12681c  1      OPC=nop             
  addq %r15, %r11              #  106   0x12681d  3      OPC=addq_r64_r64    
  jmpq %r11                    #  107   0x126820  3      OPC=jmpq_r64        
  nop                          #  108   0x126823  1      OPC=nop             
  nop                          #  109   0x126824  1      OPC=nop             
  nop                          #  110   0x126825  1      OPC=nop             
  nop                          #  111   0x126826  1      OPC=nop             
.L_126820:                     #        0x126827  0      OPC=<label>         
  addl $0x4, %edi              #  112   0x126827  3      OPC=addl_r32_imm8   
  addl $0x4, %esi              #  113   0x12682a  3      OPC=addl_r32_imm8   
  nop                          #  114   0x12682d  1      OPC=nop             
  nop                          #  115   0x12682e  1      OPC=nop             
  nop                          #  116   0x12682f  1      OPC=nop             
  nop                          #  117   0x126830  1      OPC=nop             
  nop                          #  118   0x126831  1      OPC=nop             
  nop                          #  119   0x126832  1      OPC=nop             
  nop                          #  120   0x126833  1      OPC=nop             
  nop                          #  121   0x126834  1      OPC=nop             
  nop                          #  122   0x126835  1      OPC=nop             
  nop                          #  123   0x126836  1      OPC=nop             
  nop                          #  124   0x126837  1      OPC=nop             
  nop                          #  125   0x126838  1      OPC=nop             
  nop                          #  126   0x126839  1      OPC=nop             
  nop                          #  127   0x12683a  1      OPC=nop             
  nop                          #  128   0x12683b  1      OPC=nop             
  nop                          #  129   0x12683c  1      OPC=nop             
  nop                          #  130   0x12683d  1      OPC=nop             
  nop                          #  131   0x12683e  1      OPC=nop             
  nop                          #  132   0x12683f  1      OPC=nop             
  nop                          #  133   0x126840  1      OPC=nop             
  nop                          #  134   0x126841  1      OPC=nop             
  nop                          #  135   0x126842  1      OPC=nop             
  nop                          #  136   0x126843  1      OPC=nop             
  nop                          #  137   0x126844  1      OPC=nop             
  nop                          #  138   0x126845  1      OPC=nop             
  nop                          #  139   0x126846  1      OPC=nop             
.L_126840:                     #        0x126847  0      OPC=<label>         
  movl %edi, %edi              #  140   0x126847  2      OPC=movl_r32_r32    
  movl (%r15,%rdi,1), %eax     #  141   0x126849  4      OPC=movl_r32_m32    
  movl %esi, %esi              #  142   0x12684d  2      OPC=movl_r32_r32    
  cmpl (%r15,%rsi,1), %eax     #  143   0x12684f  4      OPC=cmpl_r32_m32    
  jne .L_1267a0                #  144   0x126853  6      OPC=jne_label_1     
  leal -0x1010101(%rax), %edx  #  145   0x126859  6      OPC=leal_r32_m16    
  notl %eax                    #  146   0x12685f  2      OPC=notl_r32        
  andl %eax, %edx              #  147   0x126861  2      OPC=andl_r32_r32    
  nop                          #  148   0x126863  1      OPC=nop             
  nop                          #  149   0x126864  1      OPC=nop             
  nop                          #  150   0x126865  1      OPC=nop             
  nop                          #  151   0x126866  1      OPC=nop             
  andl $0x80808080, %edx       #  152   0x126867  6      OPC=andl_r32_imm32  
  je .L_126820                 #  153   0x12686d  2      OPC=je_label        
  xorl %eax, %eax              #  154   0x12686f  2      OPC=xorl_r32_r32    
  popq %r11                    #  155   0x126871  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d      #  156   0x126873  7      OPC=andl_r32_imm32  
  nop                          #  157   0x12687a  1      OPC=nop             
  nop                          #  158   0x12687b  1      OPC=nop             
  nop                          #  159   0x12687c  1      OPC=nop             
  nop                          #  160   0x12687d  1      OPC=nop             
  addq %r15, %r11              #  161   0x12687e  3      OPC=addq_r64_r64    
  jmpq %r11                    #  162   0x126881  3      OPC=jmpq_r64        
  nop                          #  163   0x126884  1      OPC=nop             
  nop                          #  164   0x126885  1      OPC=nop             
  nop                          #  165   0x126886  1      OPC=nop             
  nop                          #  166   0x126887  1      OPC=nop             
  nop                          #  167   0x126888  1      OPC=nop             
  nop                          #  168   0x126889  1      OPC=nop             
  nop                          #  169   0x12688a  1      OPC=nop             
  nop                          #  170   0x12688b  1      OPC=nop             
  nop                          #  171   0x12688c  1      OPC=nop             
  nop                          #  172   0x12688d  1      OPC=nop             
                                                                             
.size strcmp, .-strcmp

