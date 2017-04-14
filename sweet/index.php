<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>THE SWEET SHOP</title>
    <link rel="shortcut icon" type="image/png" href="https://pbs.twimg.com/profile_images/647762978050150401/IjRPx84i.png"/>
    <link href="boostrap/css/bootstrap.min.css" rel="stylesheet">
    <style>
    body {
   		 background-image: url("image/sweet.jpg");
   		 background-opacity: .5;
    	 background-repeat: no-repeat;
    	-webkit-background-size: cover;
    	
			}
	.row .well{
		/*background-color:#FAEBD7;*/
		font-size: 15px;
	}

	.navbar-brand{
		font-size: 35px;

	}
	#header {
		height:30%;
		color:#FAEBD7;


	}
	</style>
  </head>

  <body>

    <div id="header"><!-- Static navbar -->
    <div class="navbar navbar-default navbar-static-top">
      <div class="container">
        <div class="navbar-header">
          <a class="navbar-brand" href="index.php">THE SWEET SHOP </a>
        </div>
      </div>
    </div>
</div>

    <div class="container">
    
    	<div class="row">
	       <?php 
	       	//scan "uploads" folder and display them accordingly
	       $folder = "uploads";
	       $results = scandir('uploads');
	       foreach ($results as $result) {
	       	if ($result === '.' or $result === '..') continue;
	       
	       	if (is_file($folder . '/' . $result)) {
	       		echo '
	       		<div class="col-md-3">
		       		<div class="thumbnail">
			       		<img src="'.$folder . '/' . $result.'" alt="...">
				       		<div class="caption">
				       		<p><a href="remove.php?name='.$result.'" class="btn btn-danger btn-xs" role="button">Remove</a></p>
			       		</div>
		       		</div>
	       		</div>';
	       	}
	       }
	       ?>
    	</div>
    	
		

	      <div class="row">
	      	<div class="col-lg-12">
	           <form class="well" action="upload.php" method="post" enctype="multipart/form-data">
				  <div class="form-group">
				    <label for="file">Select a file to upload</label>
				    <input type="file" name="file">
				    <p class="help-block">Only jpg,jpeg,png and gif file with maximum size of 1 MB is allowed.</p>
				  </div>
				  <input type="submit" class="btn btn-lg btn-primary" value="Upload">
				</form>
			</div>
	      </div>
    </div> <!-- /container -->

  </body>
</html>